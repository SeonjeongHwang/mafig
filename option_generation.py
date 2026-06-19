from builtins import print
import os, time
import argparse
import copy

from utils.evaluators import check_option_constraints, Lexi_nltk, OptionNeutralityEvaluator, FactualityEvaluator, Propositionalizer, ComplexityEvaluator
from utils.examples import build_generated_passage_option_examples, build_human_passage_option_examples, normalize_passage_data_list
from utils.io import read_json, restore_examples_trajectory, write_json, write_marker
from utils.results import build_option_result, select_one_result_per_source, split_results_by_success
from utils.runtime import MODEL_NICKNAME_TO_NAME, initialize_model as initialize_vllm_model, release_model, set_random_seed

args = None
used_constraints = ["factuality", "evidence_scope", "transformation_level"]
args, llm, tokenizer, lex, propositionalizer, neutrality_evaluator, factuality_evaluator, complexity_evaluator = None, None, None, None, None, None, None, None
current_model_name = None

def parse_args():
    parser = argparse.ArgumentParser(description="DCAQG Pipeline Option Generation Arguments")
    parser.add_argument("--seed", type=int, default=2025, help="Random seed for reproducibility")
    parser.add_argument("--model_nickname", type=str, required=True, help="LLM model nickname")
    parser.add_argument("--run_name", type=str, default="dev", help="Run name for logging")
    parser.add_argument("--passage_file", type=str, default="data/ReCo.test.passage.dev.json", help="Path to the input data file")
    parser.add_argument("--output_dir", type=str, default="output/option_generation", help="Directory to save the output")
    parser.add_argument("--from_human_passage", action='store_true', help="If true, use human-written passages from the input file")
    parser.add_argument("--previous_success_file", type=str, default=None, help="Path to a previous file to continue from")
    
    parser.add_argument("--only_draft", action='store_true', help="If true, only perform drafting without revision")
    parser.add_argument("--drafter_n", type=int, default=1, help="Number of samples to draft for each example")
    parser.add_argument("--drafter_top_k", type=int, default=20, help="Top-k sampling for drafter")
    parser.add_argument("--drafter_top_p", type=float, default=0.8, help="Top-p sampling for drafter")
    parser.add_argument("--drafter_temperature", type=float, default=0.7, help="Temperature for drafter")
    parser.add_argument("--drafter_max_attempts", type=int, default=5, help="Maximum attempts for drafter to generate valid output")
    parser.add_argument("--drafter_max_tokens", type=int, default=3000, help="Maximum tokens for drafter to generate")
    parser.add_argument("--drafter_use_exemplars", action='store_true', help="If true, use exemplars in drafting")
    parser.add_argument("--already_drafted", action='store_true', help="If true, use already drafted passages from 'drafted_passages.json'")
    
    ### Revision Agents
    parser.add_argument("--continue_revision", action='store_true', help="If true, continue revision from the last saved state")
    parser.add_argument("--already_revised", action='store_true', help="If true, use already revised passages from 'all_results.json'")
    parser.add_argument("--revision_max_round", type=int, default=20, help="Maximum number of rounds for passage revision")
        
    parser.add_argument("--planner_top_k", type=int, default=20, help="Top-k sampling for planner")
    parser.add_argument("--planner_top_p", type=float, default=0.8, help="Top-p sampling for planner")
    parser.add_argument("--planner_temperature", type=float, default=0.7, help="Temperature for planner")
    parser.add_argument("--planner_max_attempts", type=int, default=5, help="Maximum attempts for planner to generate valid output")
    parser.add_argument("--planner_max_tokens", type=int, default=3000, help="Maximum tokens for planner to generate")
    
    ### Ablation
    parser.add_argument("--off_creativity_enhancement", action='store_true', help="If true, turn off creativity enhancement in agents")
    parser.add_argument("--off_planner_instruction", action='store_true', help="If true, turn off planner instructions in agents")
    parser.add_argument("--off_reworder_message", action='store_true', help="If true, turn off reworder messages in agents")
    
    parser.add_argument("--reviser_top_k", type=int, default=20, help="Top-k sampling for reviser")
    parser.add_argument("--reviser_top_p", type=float, default=0.8, help="Top-p sampling for reviser")
    parser.add_argument("--reviser_temperature", type=float, default=0.7, help="Temperature for reviser")
    parser.add_argument("--reviser_max_attempts", type=int, default=5, help="Maximum attempts for reviser to generate valid output")
    parser.add_argument("--reviser_max_tokens", type=int, default=3000, help="Maximum tokens for reviser to generate")
    parser.add_argument("--reviser_use_exemplars", action='store_true', help="If true, use exemplars in revising")
    
    parser.add_argument("--reword_max_round", type=int, default=10)
    parser.add_argument("--reworder_top_k", type=int, default=20, help="Top-k sampling for reworder")
    parser.add_argument("--reworder_top_p", type=float, default=0.8, help="Top-p sampling for reworder")
    parser.add_argument("--reworder_temperature", type=float, default=0.7, help="Temperature for reworder")
    parser.add_argument("--reworder_max_attempts", type=int, default=5, help="Maximum attempts for reworder to generate valid output")
    parser.add_argument("--reworder_max_tokens", type=int, default=3000, help="Maximum tokens for reworder to generate")
    
    ### Refinement
    parser.add_argument("--refinement_max_round", type=int, default=5)
    parser.add_argument("--refiner_top_k", type=int, default=20, help="Top-k sampling for refiner")
    parser.add_argument("--refiner_top_p", type=float, default=0.8, help="Top-p sampling for refiner")
    parser.add_argument("--refiner_temperature", type=float, default=0.7, help="Temperature for refiner")
    parser.add_argument("--refiner_max_attempt", type=int, default=5, help="Maximum attempts for refiner to generate valid output")
    parser.add_argument("--refiner_max_tokens", type=int, default=3000, help="Maximum tokens for refiner to generate")
    
    parser.add_argument("--neutrality_evaluation_model", type=str, default="qwen3-32B", help="Model nickname for option neutrality evaluation")
    parser.add_argument("--neutrality_sampling_n", type=int, default=5, help="Number of samples for option neutrality evaluation")
    parser.add_argument("--factuality_evaluation_model", type=str, default="qwen3-32B", help="Model nickname for factuality evaluation")
    parser.add_argument("--factuality_sampling_n", type=int, default=5, help="Number of samples for factuality evaluation")
    parser.add_argument("--complexity_evaluation_model", type=str, default="qwen3-32B", help="Model nickname for evidence scope evaluation")
    parser.add_argument("--complexity_evaluation_sampling_n", type=int, default=5, help="Number of samples for reasoning complexity evaluation")
    
    return parser.parse_args()

def initialize_model(model_nickname, max_tokens=8000):
    return initialize_vllm_model(model_nickname, args.seed, max_model_len=max_tokens)

def unload_model():
    global llm

    model = llm
    llm = None
    release_model(model)

def pre_evaluate(id_list, passages, options):
    global llm, tokenizer, current_model_name
    
    FE_model = args.factuality_evaluation_model
    Cpx_model = args.complexity_evaluation_model
    Prop_model = "propositionalizer"
    
    if current_model_name != FE_model:
        unload_model()
        print(f"Switching to factuality evaluation model: {FE_model}...")
        
        llm, tokenizer = initialize_model(FE_model)
        print("LLM and Tokenizer for Factuality Evaluation initialized successfully.")
        current_model_name = FE_model
        
    id2factuality = factuality_evaluator.evaluate(llm, tokenizer, id_list, passages, options)
    factualities = [id2factuality[id] for id in id_list]
    
    if "evidence_scope" in used_constraints:
        if current_model_name != Cpx_model:
            unload_model()
            print(f"Switching to complexity evaluation (ES) model: {Cpx_model}...")
            
            llm, tokenizer = initialize_model(Cpx_model)
            print("LLM and Tokenizer for Complexity Evaluation (ES) initialized successfully.")
            current_model_name = Cpx_model
            
        id2es = complexity_evaluator.evaluate_es(llm, tokenizer, id_list, passages, options, factualities)
    else:
        id2es = dict([(id, "N/A") for id in id_list])
    
    if "transformation_level" in used_constraints:
        if current_model_name != Cpx_model:
            unload_model()
            print(f"Switching to complexity evaluation (TL) model: {Cpx_model}...")
            
            llm, tokenizer = initialize_model(Cpx_model)
            print("LLM and Tokenizer for Complexity Evaluation (TL) initialized successfully.")
            current_model_name = Cpx_model
            
        id2tl = complexity_evaluator.evaluate_tl(llm, tokenizer, id_list, passages, options, factualities)
    else:
        id2tl = dict([(id, "N/A") for id in id_list])
    
    if "propositionalization" in used_constraints:
        if current_model_name != "propositionalizer":
            unload_model()
            print(f"Switching to propositionalization model: propositionalizer...")
            
            llm, tokenizer = initialize_model("propositionalizer", max_tokens=8192)
            print("LLM and Tokenizer for Propositionalization initialized successfully.")
            current_model_name = "propositionalizer"
            
        id2props = propositionalizer.get_propositions(llm, tokenizer, id_list, options)
    else:
        id2props = dict([(id, "N/A") for id in id_list])
    
    return id2factuality, id2es, id2tl, id2props
    
def write_draft(examples):
    global llm, tokenizer, current_model_name
    draft_dir = f"{args.output_dir}/{args.model_nickname}-{args.run_name}/draft"
    os.makedirs(draft_dir, exist_ok=True)
    
    if args.already_drafted or args.already_revised:
        print("Drafting options is skipped as 'draft_already_written' is set to True.")
        if os.path.exists(draft_dir):
            draft_file = os.path.join(draft_dir, "all_results.json")
            assert os.path.exists(draft_file), f"Draft file {draft_file} does not exist."
            
            id2result = read_json(draft_file)
            print(f"Loaded already drafted options from {draft_dir}.")
        else:
            print(f"No drafted options found at {draft_dir}. Proceeding to draft options.")
            return
    else:
        from utils.agents import Drafter        
                
        if current_model_name != args.model_nickname:
            unload_model()
            print(f"Switching model from {current_model_name} to {args.model_nickname} for drafting...")
            current_model_name = args.model_nickname
            llm, tokenizer = initialize_model(current_model_name)
        
        drafter = Drafter(args.model_nickname, tokenizer, "option", args.seed, args.drafter_top_k, args.drafter_top_p, args.drafter_temperature, args.drafter_max_tokens, sampling_n=args.drafter_n, use_exemplars=args.drafter_use_exemplars)
        start_time = time.time()
        id2output = drafter.draft(llm, examples, max_attempt=args.drafter_max_attempts)
        end_time = time.time()
        print(f"Option drafting completed in {(end_time - start_time) / 60:.2f} minutes.")
        
        id_list, item_passages, option_sets = [], [], []
        option_id_list, passages, options = [], [], []
        for example in examples:
            id = example["id"]
            draft_outputs = id2output[id]
            passage = example["input_data"]["passage"]
            
            if args.drafter_n == 1:
                draft_outputs = [draft_outputs]
            
            for sample_idx, draft_output in enumerate(draft_outputs):
                sample_id = f"{id}_sample{sample_idx}"
                id_list.append(sample_id)
                item_passages.append(passage)
                option_sets.append(draft_output["content"]["options"])
                
                
                for oidx in ["A", "B", "C", "D"]:
                    option_id = f"{id}_option{oidx}"
                    if args.drafter_n > 1:
                        option_id += f"_sample{sample_idx}"
                    
                    option_id_list.append(option_id)
                    passages.append(passage)
                    options.append(draft_output["content"]["options"][oidx])
                
        if current_model_name != args.neutrality_evaluation_model:
            unload_model()
            print(f"Switching to factuality evaluation model: {args.neutrality_evaluation_model}...")
            
            llm, tokenizer = initialize_model(args.neutrality_evaluation_model)
            print("LLM and Tokenizer for Neutrality Evaluation initialized successfully.")
            current_model_name = args.neutrality_evaluation_model
            
        id2neutrality = neutrality_evaluator.evaluate(llm, tokenizer, id_list, item_passages, option_sets)
        id2factuality, id2es, id2tl, id2props = pre_evaluate(option_id_list, passages, options)
        
        success_count = 0
        id2result = dict()
        for example in examples:
            id = example["id"]
            draft_outputs = id2output[id]
            
            passage = example["input_data"]["passage"]
            option_constraints = example["constraints"]["options"]
            
            if args.drafter_n == 1:
                draft_outputs = [draft_outputs]
                
            for sample_idx, draft_output in enumerate(draft_outputs):
                option_observations = dict()
                option_reports = dict()
            
                option_success = dict()
                for oidx in ["A", "B", "C", "D"]:
                    option_id = f"{id}_option{oidx}"
                    if args.drafter_n > 1:
                        option_id += f"_sample{sample_idx}"
                    
                    option = draft_output["content"]["options"][oidx]
                    temp_constraints = {"vocab_level": example["constraints"]["vocab_level"]}
                    for k, v in option_constraints[oidx].items():
                        temp_constraints[k] = v
                    
                    factuality_label = id2factuality[option_id]
                    props = id2props[option_id]
                    es_label = id2es[option_id]
                    tl_label = id2tl[option_id]

                    is_valid, report, observed_constraints = check_option_constraints(passage, option, temp_constraints, 
                                                                                    lex, factuality_label, props, es_label, tl_label, report_validity=True)
                    
                    option_observations[oidx] = observed_constraints
                    option_reports[oidx] = report
                    option_success[oidx] = is_valid
                    
                sample_id = f"{id}_sample{sample_idx}"
                neutrality = id2neutrality[sample_id]
                report, observed_constraints = dict(), dict()
                if neutrality["result"] == "acceptable":
                    report["neutrality"] = "All the options are mutually neutral."
                    observed_constraints["neutrality"] = True
                    observed_constraints["neutrality_details"] = ""
                else:
                    report["neutrality"] = "Some options are not mutually neutral. " + neutrality["reason"]
                    observed_constraints["neutrality"] = False
                    observed_constraints["neutrality_details"] = neutrality["reason"]
                    
                report["options"] = option_reports
                observed_constraints["options"] = option_observations
                        
                is_success = False
                if False not in option_success.values() and observed_constraints["neutrality"]:
                    is_success = True
                        
                ex_id = id
                if args.drafter_n > 1:
                    ex_id = ex_id + f"_sample{sample_idx}"
                        
                id2result[ex_id] = {
                    "constraints": example["constraints"],
                    "stem": draft_output["content"]["stem"],
                    "passage": passage,
                    "options": draft_output["content"]["options"],
                    "answer": draft_output["content"]["answer"],
                    "report": report,
                    "observation": observed_constraints,
                    "response": draft_output["response"],
                    "success_details": option_success,
                    "is_success": is_success
                }
                
                if is_success:
                    success_count += 1
        
        success_rate = success_count / len(examples) * 100
        print(f"Draft success rate: {success_rate:.2f}% ({success_count}/{len(examples)})")
        score_file = os.path.join(draft_dir, f"success_rate-{round(success_rate, 2)}")
        write_marker(score_file)
            
        result_file = os.path.join(draft_dir, "all_results.json")
        write_json(result_file, id2result)
            
        write_json(os.path.join(draft_dir, "args.json"), vars(args))
    
    drafted_examples = []
    for id, result in id2result.items():
        example_id = id.split("_sample")[0]
        example = copy.deepcopy([ex for ex in examples if ex["id"]==example_id][0])
        example["id"] = id
        example["trajectory"] = dict()
        example["trajectory"][0] = {
            "last_worker": "drafter",
            "drafter": {
                "response": result["response"],
                "state": {
                    "stem": result["stem"],
                    "options": result["options"],
                    "answer": result["answer"]
                },
                "observed_constraints": result["observation"],
                "report": result["report"]
            }
        }
        example["is_success"] = result["is_success"]
        example["success_details"] = result["success_details"]
        drafted_examples.append(example)
    print(f"Examples updated with drafted options.")
    
    return drafted_examples

def call_reworder(reworder, examples, round):
    ## Suggest Replacements
    
    id2output = dict()
    print("### Reworder Suggestion Phase ###")
    id2suggestion = reworder.suggest(llm, examples, round, args.reworder_max_attempts)
    for example in examples:
        id = example["id"]
        id2output[id] = {
            "suggestion_response": id2suggestion[id]["response"],
            "alternative_dict": id2suggestion[id]["output"]
        }
    
    print("Revising contents...")
    id2revision = reworder.revise(llm, examples, id2output, round, args.reworder_max_attempts)
    for example in examples:
        id = example["id"]
        id2output[id]["revision_response"] = id2revision[id]["response"]
        id2output[id]["content"] = id2revision[id]["content"]
        id2output[id]["message"] = id2revision[id]["message"]
        
    return id2output
        
def revise_option(input_examples, start_round=1):
    global llm, tokenizer, current_model_name
    from utils.agents import Planner, Reviser, Reworder
    
    revision_start_time = time.time()
    
    def update_trajectory(example, round, worker, output, id2neutrality=None, id2factuality=None, id2props=None, id2es=None, id2tl=None):
        if worker == "planner":
            example["trajectory"][round]["last_worker"] = "planner"
            example["trajectory"][round]["planner"] = {
                "response": output["response"],
                "action": output["action"],
                "target": output["target"],
                "message": output["message"]
            }
            prev_last_worker = example["trajectory"][round-1]["last_worker"]
            
            target_option = example["trajectory"][round-1][prev_last_worker]["state"]["options"][output["target"]]
            example["planner_history"].append(
                f"[Trial {round}]\n Planner decided to call {'Editor' if output['action']=='Call_Editor' else 'Reworder'} for option {output['target']}:\n{target_option}\nMessage: {output['message']}\n"
            )
        
        if worker in ["reviser", "reworder"]:
            example["trajectory"][round]["last_worker"] = worker
            
            passage = example["input_data"]["passage"]
            target_element_idx = example["trajectory"][round]["planner"]["target"]
            new_state_target = output["content"]
            constraints = example["constraints"]
            
            temp_constraints = {"vocab_level": constraints["vocab_level"]}
            target_option_constraints = constraints["options"][target_element_idx]
            for k, v in target_option_constraints.items():
                temp_constraints[k] = v
                
            ### Individual Option Check
            option_id = f"{example['id']}_option{target_element_idx}"
            factuality_label = id2factuality[option_id]
            props = id2props[option_id]
            es_label = id2es[option_id]
            tl_label = id2tl[option_id]
            
            is_valid, report_target, observed_constraints_target = check_option_constraints(passage, new_state_target, temp_constraints, 
                                                                                  lex, factuality_label, props, es_label, tl_label, report_validity=True)
            
            prev_last_worker = example["trajectory"][round-1]["last_worker"]
            new_state = copy.deepcopy(example["trajectory"][round-1][prev_last_worker]["state"])
            new_state["options"][target_element_idx] = new_state_target
            observed_constraints = copy.deepcopy(example["trajectory"][round-1][prev_last_worker]["observed_constraints"])
            observed_constraints["options"][target_element_idx] = observed_constraints_target
            report = copy.deepcopy(example["trajectory"][round-1][prev_last_worker]["report"])
            report["options"][target_element_idx] = report_target
            
            neutrality = id2neutrality[example["id"]]
            if neutrality["result"] == "acceptable":
                report["neutrality"] = "All the options are mutually neutral."
                observed_constraints["neutrality"] = True
                observed_constraints["neutrality_details"] = ""
            else:
                report["neutrality"] = "Some options are not mutually neutral. " + neutrality["reason"]
                observed_constraints["neutrality"] = False
                observed_constraints["neutrality_details"] = neutrality["reason"]
            
            if worker == "reworder":
                report["reworder_advice"] = "[Reworder's Advice]\n" + output["message"]
                
            if worker == "reviser":
                example["trajectory"][round]["reviser"] = {
                    "response": output["response"],
                    "state": new_state,
                    "observed_constraints": observed_constraints,
                    "report": report
                }
            else:
                example["trajectory"][round]["reworder"] = {
                    "response": {"suggestion": output["suggestion_response"],
                                "revision": output["revision_response"]},
                    "alternative_dict": output["alternative_dict"],
                    "state": new_state,
                    "observed_constraints": observed_constraints,
                    "report": report
                }
            example["success_details"][target_element_idx] = is_valid
            if False not in example["success_details"].values() and observed_constraints["neutrality"]:
                example["is_success"] = True
            
        return example
    
    completed_examples = []
    terminated_examples = []
    completed_id_list = []
    examples = []
    for example in input_examples:
        if example["is_success"]:
            completed_examples.append(example)
            id_prefix = example["id"].split("_sample")[0]
            completed_id_list.append(id_prefix)
            
    for example in input_examples:
        if not example["is_success"]:
            id_prefix = example["id"].split("_sample")[0]
            if id_prefix not in completed_id_list:
                examples.append(example)
            else:
                example["is_terminated"] = True
                terminated_examples.append(example)
        
    revision_dir = f"{args.output_dir}/{args.model_nickname}-{args.run_name}/revision-{args.revision_max_round}R"
    os.makedirs(revision_dir, exist_ok=True)
    
    if current_model_name != args.model_nickname:
        unload_model()
        print(f"Switching model from {current_model_name} to {args.model_nickname}...")
        
        current_model_name = args.model_nickname
        llm, tokenizer = initialize_model(current_model_name)
        print("LLM and Tokenizer for Revision successfully.")
        
    planner = Planner(args.model_nickname, tokenizer, "option", args.seed, args.planner_top_k, args.planner_top_p, args.planner_temperature, args.planner_max_tokens, args=args)
    reviser = Reviser(args.model_nickname, tokenizer, "option", args.seed, args.reviser_top_k, args.reviser_top_p, args.reviser_temperature, args.reviser_max_tokens, args.reviser_use_exemplars, args=args)
    reworder = Reworder(args.model_nickname, tokenizer, lex, "option", args.seed, args.reworder_top_k, args.reworder_top_p, args.reworder_temperature, args.reworder_max_tokens, args=args)
    
    for r in range(start_round, args.revision_max_round+1):
        for example in examples:
            example["trajectory"][r] = {"last_worker": None}
            
        print(f"### Revision Round-{r} | {len(examples)} examples ###")
        
        if current_model_name != args.model_nickname:
            unload_model()
            print(f"Switching model from {current_model_name} to {args.model_nickname}...")
            
            current_model_name = args.model_nickname
            llm, tokenizer = initialize_model(current_model_name)
            print("LLM and Tokenizer for Revision successfully.")
        
        planner_outputs = planner.call(llm, examples, r, args.planner_max_attempts)
        
        reviser_loads = []
        reworder_loads = []
        for example in examples:
            id = example["id"]
            planner_output = planner_outputs[id]
            example = update_trajectory(example, r, "planner", planner_output)
            
            if planner_output["action"] == "Call_Editor":
                reviser_loads.append(example)
            elif planner_output["action"] == "Call_Reworder":
                reworder_loads.append(example)
        
        if len(reviser_loads)+len(reworder_loads) == 0:
            print("All examples are successfully revised. Exiting revision loop.")
            examples = []
            break
        
        ### Call Agents
        reviser_outputs, reworder_outputs = dict(), dict()
        if reviser_loads:
            reviser_outputs = reviser.call(llm, reviser_loads, r, args.reviser_max_attempts) ## str
        if reworder_loads:
            reworder_outputs = call_reworder(reworder, reworder_loads, r) ## str
            
        id_list, item_passages, option_sets = [], [], []
        option_id_list, passages, options = [], [], []
        for example in reviser_loads + reworder_loads:
            example_id = example["id"]
            if example_id in reviser_outputs:
                new_option = reviser_outputs[example_id]["content"]
            else:
                new_option = reworder_outputs[example_id]["content"]
            
            passage = example["input_data"]["passage"]
            option_id = f"{example['id']}_option{example['trajectory'][r]['planner']['target']}"
            
            new_state = copy.deepcopy(example["trajectory"][r-1][example["trajectory"][r-1]["last_worker"]]["state"]["options"])
            target_id = example["trajectory"][r]["planner"]["target"]
            new_state[target_id] = new_option
            
            id_list.append(example["id"])
            item_passages.append(passage)
            option_sets.append(new_state)
            
            option_id_list.append(option_id)
            passages.append(passage)
            options.append(new_option)
            
        if current_model_name != args.neutrality_evaluation_model:
            unload_model()
            print(f"Switching to factuality evaluation model: {args.neutrality_evaluation_model}...")
            
            llm, tokenizer = initialize_model(args.neutrality_evaluation_model)
            print("LLM and Tokenizer for Neutrality Evaluation initialized successfully.")
            current_model_name = args.neutrality_evaluation_model
            
        id2neutrality = neutrality_evaluator.evaluate(llm, tokenizer, id_list, item_passages, option_sets)
        id2factuality, id2es, id2tl, id2props = pre_evaluate(option_id_list, passages, options)
        
        remain_examples = []
        ### Update Trajectory
        for example in reviser_loads:
            id = example["id"]
            reviser_output = reviser_outputs[id]
            example = update_trajectory(example, r, "reviser", reviser_output, id2neutrality, id2factuality, id2props, id2es, id2tl)
            if example["is_success"]:
                completed_examples.append(example)
                id_prefix = example["id"].split("_sample")[0]
                completed_id_list.append(id_prefix)
            else:
                remain_examples.append(example)
            
        for example in reworder_loads:
            id = example["id"]
            reworder_output = reworder_outputs[id]
            example = update_trajectory(example, r, "reworder", reworder_output, id2neutrality, id2factuality, id2props, id2es, id2tl)
            if example["is_success"]:
                completed_examples.append(example)
                id_prefix = example["id"].split("_sample")[0]
                completed_id_list.append(id_prefix)
            else:
                remain_examples.append(example)
                
        examples = []
        for example in remain_examples:
            id_prefix = example["id"].split("_sample")[0]
            if id_prefix not in completed_id_list:
                examples.append(example)
            else:
                example["is_terminated"] = True
                terminated_examples.append(example)
        
        print(f"### Revision Round-{r} Completed ###")
        print(f"# Success / # Terminated / # Retry Needed : {len(completed_examples)} / {len(terminated_examples)} / {len(examples)} (unique: {len(set(ex['id'].split('_sample')[0] for ex in examples))})")
        
        log_file = os.path.join(revision_dir, "revision_history.json")
        write_json(log_file, completed_examples + terminated_examples + examples, indent=3)
        print(f"Revised passages saved to {log_file}.")
        
    print(f"# Success / # Terminated / # Retry Needed : {len(completed_examples)} / {len(terminated_examples)} / {len(examples)} (unique: {len(set(ex['id'].split('_sample')[0] for ex in examples))})")
    all_examples = completed_examples + terminated_examples + examples
        
    success_rate = round(len(completed_examples)/len(all_examples)*100, 2)
    print(f"Revision Success Rate: {success_rate:.2f}% ({len(completed_examples)}/{len(set(ex['id'].split('_sample')[0] for ex in all_examples))})")
    score_file = os.path.join(revision_dir, f"success_rate-{round(success_rate, 2)}")
    write_marker(score_file)
        
    revision_end_time = time.time()
    print(f"Option revision completed in {(revision_end_time - revision_start_time) / 60:.2f} minutes.") 
        
    all_results, success_results, fail_results = split_results_by_success(all_examples, build_option_result)
        
    write_json(os.path.join(revision_dir, "success_results.json"), success_results)
        
    write_json(os.path.join(revision_dir, "fail_results.json"), fail_results)
        
    write_json(os.path.join(revision_dir, "all_results.json"), all_results)
        
    write_json(os.path.join(revision_dir, "args.json"), vars(args))
        
    log_file = os.path.join(revision_dir, "revision_history.json")
    write_json(log_file, all_examples, indent=3)
    print(f"Revised options saved to {log_file}.")
    
    return all_examples
    
def refinement(input_examples):
    examples = input_examples[:]
    global llm, tokenizer, current_model_name
    
    from utils.agents import Refiner
    
    refinement_dir = f"{args.output_dir}/{args.model_nickname}-{args.run_name}/refinement"
    os.makedirs(refinement_dir, exist_ok=True)
    
    if current_model_name != args.model_nickname:
        unload_model()
        print(f"Switching model from {current_model_name} to {args.model_nickname}...")
        
        current_model_name = args.model_nickname
        llm, tokenizer = initialize_model(current_model_name)
        print("LLM and Tokenizer for Revision successfully.")
    
    refiner = Refiner(args.model_nickname, tokenizer, "option", args.seed, args.refiner_top_k, args.refiner_top_p, args.refiner_temperature, args.refiner_max_tokens)
    refined_outputs = refiner.refine(llm, examples, args.refiner_max_attempt)
    
    ## Check success rate
    id_list, item_passages, option_sets = [], [], []
    option_id_list, passages, options = [], [], []
    for example in examples:
        example_id = example["id"]
        refined_output = refined_outputs[example_id]
        
        passage = example["input_data"]["passage"]
        new_state = refined_output["options"]
        
        id_list.append(example["id"])
        item_passages.append(passage)
        option_sets.append(new_state)
        
        for oidx, new_option in new_state.items():
            option_id = f"{example['id']}_option{oidx}"
        
            option_id_list.append(option_id)
            passages.append(passage)
            options.append(new_option)
        
    if current_model_name != args.neutrality_evaluation_model:
        unload_model()
        print(f"Switching to factuality evaluation model: {args.neutrality_evaluation_model}...")
        
        llm, tokenizer = initialize_model(args.neutrality_evaluation_model)
        print("LLM and Tokenizer for Neutrality Evaluation initialized successfully.")
        current_model_name = args.neutrality_evaluation_model
        
    id2neutrality = neutrality_evaluator.evaluate(llm, tokenizer, id_list, item_passages, option_sets)
    id2factuality, id2es, id2tl, id2props = pre_evaluate(option_id_list, passages, options)
    
    success_count = 0
    refined_examples = []
    all_results = []
    for example in examples:
        id = example["id"]
        
        passage = example["input_data"]["passage"]
        options = refined_outputs[id]["options"]
        constraints = example["constraints"]
        report = dict()
        observed_constraints = dict()
        is_success = True

        neutrality = id2neutrality[id]
        if neutrality["result"] == "acceptable":
            report["neutrality"] = "All the options are mutually neutral."
            observed_constraints["neutrality"] = True
            observed_constraints["neutrality_details"] = ""
        else:
            report["neutrality"] = "Some options are not mutually neutral. " + neutrality["reason"]
            observed_constraints["neutrality"] = False
            observed_constraints["neutrality_details"] = neutrality["reason"]
            is_success = False
            
        
        report["options"] = dict()
        observed_constraints["options"] = dict()
        for oidx, option in options.items():
            option_id = f"{example['id']}_option{oidx[0]}"
            temp_constraints = {"vocab_level": constraints["vocab_level"]}
            for k, v in constraints["options"][oidx].items():
                temp_constraints[k] = v
                
            factuality_label = id2factuality[option_id]
            props = id2props[option_id]
            es_label = id2es[option_id]
            tl_label = id2tl[option_id]            
        
            is_valid, report_option, observed_constraints_option = check_option_constraints(passage, option, temp_constraints, 
                                                                                    lex, factuality_label, props, es_label, tl_label, report_validity=True)
            
            if is_valid is False:
                is_success = False
                
            report["options"][oidx] = report_option
            observed_constraints["options"][oidx] = observed_constraints_option
            
        prev_last_worker = example["trajectory"][max(example["trajectory"].keys())]["last_worker"]
        last_state = example["trajectory"][max(example["trajectory"].keys())][prev_last_worker]["state"]
        new_state = copy.deepcopy(last_state)
        new_state["options"] = options
            
        new_trajectory = dict()
        new_trajectory[0] = {
            "last_worker": "refiner",
            "refiner": {
                "response": refined_outputs[id]["response"],
                "state": new_state,
                "observed_constraints": observed_constraints,
                "report": report
            }
        }
        example["trajectory"] = new_trajectory
        example["planner_history"] = []
        example["is_success"] = is_success
        refined_examples.append(example)
        
        if is_success:
            success_count += 1
    
        result = {"id": id,
                  "source_id": example["source_id"],
                  "level": example["level"],
                  "passage": example["input_data"]["passage"],
                  "constraints": example["constraints"],
                  "stem": new_state["stem"],
                  "options": new_state["options"],
                  "answer": new_state["answer"]}
        all_results.append(result)
        
    success_rate = success_count / len(refined_examples) * 100
    print(f"Refinement success rate: {success_rate:.2f}% ({success_count}/{len(refined_examples)})")
    score_file = os.path.join(refinement_dir, f"success_rate-{round(success_rate, 2)}")
    write_marker(score_file)
        
    result_file = os.path.join(refinement_dir, "all_results.json")
    write_json(result_file, all_results)
    print(f"Refinement results saved to {result_file}.")
    
    write_json(os.path.join(refinement_dir, "args.json"), vars(args))
        
    log_file = os.path.join(refinement_dir, "revision_history.json")
    write_json(log_file, refined_examples, indent=3)
    print(f"Refined passages saved to {log_file}.")
        
    return refined_examples
    
def main():
    global args, llm, tokenizer, current_model_name, lex, neutrality_evaluator, factuality_evaluator, complexity_evaluator, propositionalizer
    args = parse_args()
    
    assert args.model_nickname in MODEL_NICKNAME_TO_NAME, f"Model nickname '{args.model_nickname}' is not recognized."
    set_random_seed(args.seed)
    
    lex = Lexi_nltk()
    neutrality_evaluator = OptionNeutralityEvaluator(args.neutrality_evaluation_model, n=args.neutrality_sampling_n)
    factuality_evaluator = FactualityEvaluator(args.factuality_evaluation_model, n=args.factuality_sampling_n)
    complexity_evaluator = ComplexityEvaluator(args.complexity_evaluation_model, n=args.complexity_evaluation_sampling_n)
    propositionalizer = Propositionalizer(args.seed)
    
    """
        Draft Option Generation
    """
    start_round = 1
    if args.continue_revision:
        revision_dir = f"{args.output_dir}/{args.model_nickname}-{args.run_name}/revision-{args.revision_max_round}R"
        log_file = os.path.join(revision_dir, "revision_history.json")
        assert os.path.exists(log_file), f"Revision history file {log_file} does not exist."
        
        saved_examples = read_json(log_file)
        examples = restore_examples_trajectory(saved_examples)
        
        start_round = max(examples[-1]["trajectory"].keys())
        print(f"Continuing edition from round {start_round}.")
        print(f"Loaded {len(examples)} examples from {log_file} for continuing edition.")
    else:
        if args.from_human_passage:
            constraint_path = "data/difficulty_series.option.easy.json"
            level2cconstraints = read_json(constraint_path)
            
            success_ids = []
            if args.previous_success_file is not None:
                success_ids = [data["id"] for data in read_json(args.previous_success_file)]
            
            examples, skipped_ids = build_human_passage_option_examples(read_json(args.passage_file), level2cconstraints, success_ids)
            for skipped_id in skipped_ids:
                print(f"Skipping already successful example: {skipped_id}")
            print(f"Loaded {len(examples)} examples | Documents from {args.passage_file} | Constraints from {constraint_path}.")
                
        else:
            ## Difficulty Level Constraints
            passage_args_file = "/".join(args.passage_file.split("/")[:-1]) + "/args.json"
            constraint_path = read_json(passage_args_file)["constraint_path"]
            level2cconstraints = read_json(constraint_path)
            
            success_ids = []
            if args.previous_success_file is not None:
                success_ids = [data["id"] for data in read_json(args.previous_success_file)]
            
            passage_data_list, detected_dict = normalize_passage_data_list(read_json(args.passage_file))
            if detected_dict:
                print("Detected dictionary format for passage data. Converting to list format.")
            examples, skipped_ids = build_generated_passage_option_examples(passage_data_list, level2cconstraints, success_ids)
            for skipped_id in skipped_ids:
                print(f"Skipping already successful example: {skipped_id}")
            print(f"Loaded {len(examples)} examples | Documents from {args.passage_file} | Constraints from {constraint_path}.")
    
        examples = write_draft(examples)
        
    if args.only_draft:
        print("Only drafting is performed as 'only_draft' is set to True. Exiting.")
        return
        
    if args.already_revised:
        print("Skipping revision as 'already_revised' is set to True.")
        revision_file = f"{args.output_dir}/{args.model_nickname}-{args.run_name}/revision-{args.revision_max_round}R/revision_history.json"
        examples = restore_examples_trajectory(read_json(revision_file))
            
        print(f"Loaded {len(examples)} revised examples from {revision_file}.")
    else:
        examples = revise_option(examples, start_round)
    
    if args.drafter_n > 1:
        final_results = select_one_result_per_source([
            build_option_result(example, include_status=True)
            for example in examples
        ])
        final_file_name = f"{args.output_dir}/{args.model_nickname}-{args.run_name}/final_results.json"
        write_json(final_file_name, final_results)
        
if __name__ == "__main__":
    main()
