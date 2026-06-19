from builtins import print
import os, time
import argparse
import copy

from utils.evaluators import Lexi_nltk, check_passage_constraints
from utils.agents import Planner, Reviser, Reworder, Refiner
from utils.examples import build_passage_examples
from utils.io import read_json, restore_examples_trajectory, write_json, write_marker
from utils.results import build_passage_result, select_one_result_per_source, split_results_by_success
from utils.runtime import MODEL_NICKNAME_TO_NAME, initialize_model, set_random_seed

args = None
llm = None
tokenizer = None
lex = None

def parse_args():
    parser = argparse.ArgumentParser(description="DCAQG Pipeline Passage Generation Arguments")
    parser.add_argument("--seed", type=int, default=2025, help="Random seed for reproducibility")
    parser.add_argument("--model_nickname", type=str, required=True, help="LLM model nickname")
    parser.add_argument("--run_name", type=str, default="dev", help="Run name for logging")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the input data file")
    parser.add_argument("--constraint_path", type=str, default="data/constraints.passage.json", help="Path to the constraint combinations file")
    parser.add_argument("--output_dir", type=str, default="output/passage_generation", help="Directory to save the output")
    parser.add_argument("--previous_success_file", type=str, default=None, help="Path to a previous file to continue from")
    
    parser.add_argument("--only_draft", action='store_true', help="If true, only perform drafting without revision")
    parser.add_argument("--drafter_n", type=int, default=1, help="Number of samples to draft for each example")
    parser.add_argument("--drafter_top_k", type=int, default=20, help="Top-k sampling for drafter")
    parser.add_argument("--drafter_top_p", type=float, default=0.8, help="Top-p sampling for drafter")
    parser.add_argument("--drafter_temperature", type=float, default=0.7, help="Temperature for drafter")
    parser.add_argument("--drafter_max_attempts", type=int, default=5, help="Maximum attempts for drafter to generate valid output")
    parser.add_argument("--drafter_max_tokens", type=int, default=3000, help="Maximum tokens for drafter to generate")
    parser.add_argument("--already_drafted", action='store_true', help="If true, use already drafted passages from 'drafted_passages.json'")
    
    ### Revision Agents
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
    
    return parser.parse_args()

def write_draft(examples):
    draft_dir = f"{args.output_dir}/{args.model_nickname}-{args.run_name}/draft"
    os.makedirs(draft_dir, exist_ok=True)
    draft_file = os.path.join(draft_dir, "all_results.json")
    
    ## If already drafted passages exist, load them
    if args.already_drafted or args.already_revised:
        print("Drafting passages is skipped as 'already_drafted' is set to True.")
        if os.path.exists(draft_dir):
            assert os.path.exists(draft_file), f"Draft file {draft_file} does not exist."
            
            id2result = read_json(draft_file)
            print(f"Loaded already drafted results from {draft_file}.")
        else:
            print(f"No drafted passages found at {draft_dir}. Proceeding to draft passages.")
            return
    else:
        from utils.agents import Drafter
        
        drafter = Drafter(args.model_nickname, tokenizer, "passage", args.seed, args.drafter_top_k, args.drafter_top_p, args.drafter_temperature, args.drafter_max_tokens, args.drafter_n)    
        print("Starting passage drafting...")
        start_time = time.time()
        id2output = drafter.draft(llm, examples, args.drafter_max_attempts) ## return {"response": ..., "content": string or null}
        end_time = time.time()
        print(f"Passage drafting completed in {(end_time - start_time) / 60:.2f} minutes.")
        
        num_error = 0
        id2result = dict()
        for ex_id, outputs in id2output.items():
            example = [ex for ex in examples if ex["id"] == ex_id][0]
            constraints = example["constraints"]
            
            if args.drafter_n == 1:
                outputs = [outputs]
                
            for sample_idx, output in enumerate(outputs):
                id = ex_id
                if args.drafter_n > 1:
                    id = f"{id}_sample{sample_idx}"
            
                id2result[id] = {
                    "constraints": constraints,
                    "passage": [],
                    "report": "",
                    "observation": None,
                    "response": output["response"],
                    "is_success": False
                }
                
                if output["content"] is None:
                    del id2result[id]
                    num_error += 1
                    continue
                
                passage = output["content"]
                
                id2result[id]["passage"] = passage
                is_valid, report, observed_constraints = check_passage_constraints(passage, constraints, lex)
                
                id2result[id]["report"] = report
                id2result[id]["observation"] = observed_constraints
                id2result[id]["is_success"] = is_valid
                
        success_id_list = []
        for id, res in id2result.items():
            if res["is_success"]:
                success_id_list.append(id.split("_sample")[0])
        success_count = len(set(success_id_list))
        total_id_list = [ex["id"] for ex in examples]
        
        success_rate = success_count / len(total_id_list) * 100
        print(f"# Error: {num_error}")
        print(f"Draft success rate: {success_rate:.2f}% ({success_count}/{len(total_id_list)})")
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
        example["trajectory"][0] = {
            "last_worker": "drafter",
            "drafter": {
                "response": result["response"],
                "state": result["passage"],
                "observed_constraints": result["observation"],
                "report": result["report"]
            }
        }
        example["is_success"] = result["is_success"]
        drafted_examples.append(example)
    print(f"Examples updated with drafted passages.")
    
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
        
    ## Revise Contents
    print("Revising contents...")
    id2revision = reworder.revise(llm, examples, id2output, round, args.reworder_max_attempts)
    for example in examples:
        id = example["id"]
        id2output[id]["revision_response"] = id2revision[id]["response"]
        id2output[id]["content"] = id2revision[id]["content"]
        id2output[id]["message"] = id2revision[id]["message"]
    
    return id2output
        
def revise_passage(input_examples, revision_step=1):
    
    def update_trajectory(example, round, worker, output):
        if worker == "planner":
            example["trajectory"][round]["last_worker"] = "planner"
            example["trajectory"][round]["planner"] = {
                "response": output["response"],
                "action": output["action"],
                "message": output["message"]
            }
            example["planner_history"].append(
                f"[Trial {round}]\n Planner decided to call {'Editor' if output['action']=='Call_Editor' else 'Reworder'}\nMessage: {output['message']}\n"
            )
            
        if worker == "reviser":
            example["trajectory"][round]["last_worker"] = "reviser"
            
            new_state = output["content"]
            constraints = example["constraints"]
            
            is_valid, report, observed_constraints = check_passage_constraints(new_state, constraints, lex)
            example["trajectory"][round]["reviser"] = {
                "response": output["response"],
                "state": new_state,
                "observed_constraints": observed_constraints,
                "report": report
            }
            if is_valid:
                example["is_success"] = True     
        
        if worker == "reworder":
            example["trajectory"][round]["last_worker"] = "reworder"
            
            new_state = output["content"]
            constraints = example["constraints"]
            
            is_valid, report, observed_constraints = check_passage_constraints(new_state, constraints, lex)
            
            report["reworder_advice"] = "[Reworder's Advice]\n" + output["message"]
            example["trajectory"][round]["reworder"] = {
                "response": {"suggestion": output["suggestion_response"],
                             "revision": output["revision_response"]},
                "alternative_dict": output["alternative_dict"],
                "state": new_state,
                "observed_constraints": observed_constraints,
                "report": report
            }
            if is_valid:
                example["is_success"] = True 
            
        return example
    
    revision_dir = f"{args.output_dir}/{args.model_nickname}-{args.run_name}/revision-{args.revision_max_round}R-step{revision_step}"
    os.makedirs(revision_dir, exist_ok=True)
    
    planner = Planner(args.model_nickname, tokenizer, "passage", args.seed, args.planner_top_k, args.planner_top_p, args.planner_temperature, args.planner_max_tokens, args=args)
    reviser = Reviser(args.model_nickname, tokenizer, "passage", args.seed, args.reviser_top_k, args.reviser_top_p, args.reviser_temperature, args.reviser_max_tokens, args=args)
    reworder = Reworder(args.model_nickname, tokenizer, lex, "passage", args.seed, args.reworder_top_k, args.reworder_top_p, args.reworder_temperature, args.reworder_max_tokens, args=args)
    
    completed_examples = []
    completed_id_list = []
    terminated_examples = []
    remain_examples = []
    for example in input_examples:
        if example["is_success"]:
            completed_examples.append(example)
            completed_id_list.append(example["id"].split("_sample")[0])
        else:
            remain_examples.append(example)
            
    examples = []
    for example in remain_examples:
        if example["id"].split("_sample")[0] in completed_id_list:
            example["is_terminated"] = True
            terminated_examples.append(example)
        else:
            examples.append(example)
            
    for r in range(1, args.revision_max_round+1):
        for example in examples:
            example["trajectory"][r] = {"last_worker": None}
            
        print(f"### Revision Round-{r} | {len(examples)} examples ###")
        
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
            else:
                raise ValueError(f"Unknown action {planner_output['action']} from planner.")
                
        if len(reviser_loads)+len(reworder_loads) == 0:
            print("All examples are successfully revised. Exiting revision loop.")
            examples = []
            break
                
        ### Call Agents
        if reviser_loads:
            reviser_outputs = reviser.call(llm, reviser_loads, r, args.reviser_max_attempts)
        if reworder_loads:
            reworder_outputs = call_reworder(reworder, reworder_loads, r)
            
        remain_examples = []
        ### Update Trajectory
        for example in reviser_loads:
            id = example["id"]
            reviser_output = reviser_outputs[id]
            example = update_trajectory(example, r, "reviser", reviser_output)
            if example["is_success"]:
                completed_examples.append(example)
                completed_id_list.append(example["id"].split("_sample")[0])
            else:
                remain_examples.append(example)
            
        for example in reworder_loads:
            id = example["id"]
            reworder_output = reworder_outputs[id]
            example = update_trajectory(example, r, "reworder", reworder_output)
            if example["is_success"]:
                completed_examples.append(example)
                completed_id_list.append(example["id"].split("_sample")[0])
            else:
                remain_examples.append(example)
                
        examples = []
        for example in remain_examples:
            if example["id"].split("_sample")[0] in completed_id_list:
                example["is_terminated"] = True
                terminated_examples.append(example)
            else:
                examples.append(example)
            
        print(f"### Revision Round-{r} Completed ###")
        print(f"# Success / # Terminated / # Retry Needed : {len(completed_examples)} / {len(terminated_examples)} / {len(examples)} (unique: {len(set(ex['id'].split('_sample')[0] for ex in examples))})")
        
        if len(examples) == 0:
            print("All examples are successfully revised. Exiting revision loop.")
            break
        
        log_file = os.path.join(revision_dir, "revision_history.json")
        write_json(log_file, completed_examples + terminated_examples + examples, indent=3)
        print(f"Revised passages saved to {log_file}.")
        
    all_examples = completed_examples + terminated_examples + examples
    print(f"# Success / # Terminated / # Retry Needed : {len(completed_examples)} / {len(terminated_examples)} / {len(examples)} (unique: {len(set(ex['id'].split('_sample')[0] for ex in examples))})")
            
    ## Check success rate
    completed_unique_ids = set(ex['id'].split('_sample')[0] for ex in completed_examples)
    all_unique_ids = set(ex['id'].split('_sample')[0] for ex in all_examples)
    
    success_rate = round(len(completed_unique_ids)/len(all_unique_ids)*100, 2)
    print(f"Revision Success Rate: {success_rate:.2f}% ({len(completed_unique_ids)/len(all_unique_ids)})")
    score_file = os.path.join(revision_dir, f"success_rate-{round(success_rate, 2)}")
    write_marker(score_file)
        
    all_results, success_results, fail_results = split_results_by_success(all_examples, build_passage_result)
        
    write_json(os.path.join(revision_dir, "success_results.json"), success_results)
        
    write_json(os.path.join(revision_dir, "fail_results.json"), fail_results)
        
    write_json(os.path.join(revision_dir, "all_results.json"), all_results)
        
    write_json(os.path.join(revision_dir, "args.json"), vars(args))
        
    log_file = os.path.join(revision_dir, "revision_history.json")
    write_json(log_file, all_examples, indent=3)
    print(f"Revised passages saved to {log_file}.")
    
    return all_examples

def refinement(input_examples, step):
    examples = input_examples[:]
    
    refiner = Refiner(args.model_nickname, tokenizer, "passage", args.seed, args.refiner_top_k, args.refiner_top_p, args.refiner_temperature, args.refiner_max_tokens)
    refined_outputs = refiner.refine(llm, examples, args.refiner_max_attempt)
    
    refinement_dir = f"{args.output_dir}/{args.model_nickname}-{args.run_name}/refinement-step{step}"
    os.makedirs(refinement_dir, exist_ok=True)
    
    id2results = dict()
    ## Check success rate
    success_id_list = []
    num_error = 0
    
    all_results = []
    refined_examples = []
    for example in examples:
        id = example["id"]
        
        passage = refined_outputs[id]["passage"]
        constraints = example["constraints"]
        is_valid, report, observed_constraint = check_passage_constraints(passage, constraints, lex)
        
        id2results[id] = {"constraints": constraints,
                          "report": report,
                          "observation": observed_constraint,
                          "response": refined_outputs[id]["response"]}
        
        new_trajectory = dict()
        new_trajectory[0] = {
            "last_worker": "refiner",
            "refiner": {
                "response": refined_outputs[id]["response"],
                "state": passage,
                "observed_constraints": observed_constraint,
                "report": report
            }
        }
        example["trajectory"] = new_trajectory
        example["planner_history"] = []
        example["is_success"] = is_valid
        refined_examples.append(example)
        
        if is_valid:
            success_id_list.append(id)
    
        result = {"id": id,
                  "source_id": example["source_id"],
                  "level": example["level"],
                  "source_text": example["input_data"]["source_text"],
                  "constraints": example["constraints"],
                  "passage": passage}
        all_results.append(result)
    
    success_count = len(set([ex.split("_sample")[0] for ex in success_id_list]))
    total_count = len(set([ex["id"].split("_sample")[0] for ex in refined_examples]))
    success_rate = success_count / total_count * 100
    print(f"# Error: {num_error}")
    print(f"Refinement success rate: {success_rate:.2f}% ({success_count}/{total_count})")
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
    global args, llm, tokenizer, lex
    args = parse_args()
    
    assert args.model_nickname in MODEL_NICKNAME_TO_NAME, f"Model nickname '{args.model_nickname}' is not recognized."
        
    set_random_seed(args.seed)
    llm, tokenizer = initialize_model(args.model_nickname, args.seed)
    print("LLM and Tokenizer initialized successfully.")
    print(args)
    
    lex = Lexi_nltk()
    
    already_success_ids = []
    if args.previous_success_file is not None:
        assert os.path.exists(args.previous_success_file), f"Previous file {args.previous_success_file} does not exist."
        previous_results = read_json(args.previous_success_file)
        already_success_ids = [res["id"] for res in previous_results]
        print(f"Loaded {len(already_success_ids)} already successful example IDs from {args.previous_success_file}.")
    else:
        print("No previous file provided. Starting fresh.")
    
    combinations = read_json(args.constraint_path)
    examples, skipped_ids = build_passage_examples(read_json(args.data_path), combinations, already_success_ids)
    for skipped_id in skipped_ids:
        print(f"Skipping already successful example ID: {skipped_id}")
    print(f"Loaded {len(examples)} examples | Documents from {args.data_path} | Constraints from {args.constraint_path}.")
    
    ### Drafting
    examples = write_draft(examples)
    
    if args.only_draft:
        print("Only drafting is performed as 'only_draft' is set to True. Exiting.")
        return

    ### Revision
    completed_examples = []
    completed_id_list =  []
    terminated_examples = []
    for step in range(1, args.refinement_max_round+1):
        print(f"=== Revision + Refinement Step-{step} ===")
        if args.already_revised:
            revision_dir = f"{args.output_dir}/{args.model_nickname}-{args.run_name}/revision-{args.revision_max_round}R-step{step}"
            saved_examples = read_json(os.path.join(revision_dir, "revision_history.json"))
            examples = restore_examples_trajectory(saved_examples)
            
        else:
            examples = revise_passage(examples, step)
        examples = refinement(examples, step)
        
        remain_examples = []
        for example in examples:
            if example["is_success"]:
                completed_examples.append(example)
                completed_id_list.append(example["id"].split("_sample")[0])
            else:
                remain_examples.append(example)
                
        incomplete_examples = []
        for example in remain_examples:
            if example["id"].split("_sample")[0] in completed_id_list:
                example["is_terminated"] = True
                terminated_examples.append(example)
            else:
                incomplete_examples.append(example)
        
        if len(incomplete_examples) == 0:
            print("All examples are successfully completed. Exiting the loop.")
            break
        
        examples = incomplete_examples[:]
        
    all_results = [
        build_passage_result(example, include_status=True)
        for example in completed_examples+terminated_examples+incomplete_examples
    ]
        
    final_dir = f"{args.output_dir}/{args.model_nickname}-{args.run_name}/final-MaxStep{args.refinement_max_round}"
    os.makedirs(final_dir, exist_ok=True)
    write_json(os.path.join(final_dir, "all_results.json"), all_results)
    print(f"Final results saved to {final_dir}/all_results.json")
    
    success_cnt = len(set([ex['id'].split('_sample')[0] for ex in completed_examples]))
    total_cnt = len(set([ex["id"].split("_sample")[0] for ex in all_results]))
    success_rate = round(success_cnt/total_cnt*100, 2)
    print(f"Final Success Rate: {success_rate}% ({success_cnt}/{total_cnt})")
    success_file = os.path.join(final_dir, f"success_rate-{success_rate}")
    write_marker(success_file)
        
    write_json(os.path.join(final_dir, "args.json"), vars(args))
        
    if args.drafter_n > 1:
        final_results = select_one_result_per_source(all_results, drop_keys=("is_success", "is_terminated"))
        write_json(os.path.join(final_dir, "all_results.json"), final_results)
    
if __name__ == "__main__":
    main()
