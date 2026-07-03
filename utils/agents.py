import json, re, copy
from vllm import SamplingParams
import nltk, torch

LOG_ON = [] #["planner", "editor", "drafter", "fact_selector", "reworder", "refiner"]

def extract_json(response):
    match = re.search(r"```json\s*({.*?})\s*```", response, re.DOTALL)
    if match:
        json_str = match.group(1)
        try:
            data = json.loads(json_str)
            return data
        except json.JSONDecodeError:
            return None
    return None

def sentence_segment(text):
    """
    Segment the text into sentences using nltk.
    """
    
    temp_sents = [sent.strip() for sent in nltk.sent_tokenize(text)]
    sentences = []
    for sent in temp_sents:
        if len(sentences) == 0:
            sentences.append(sent)
        else:
            if len(sent) == 0:
                continue
            elif sent[0].isalpha() and sent[0].islower():
                sentences[-1] += " " + sent
            else:
                sentences.append(sent)
    
    return sentences

def get_chat_template(tokenizer, model_nickname, input_prompt, is_reasoning):
    if model_nickname in ["gemma3-27B"]:
        chat_example = tokenizer.apply_chat_template(
            [{"role": "user", "content": [{"type": "text", "text": input_prompt}]}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=is_reasoning
        )        
        
    else:
        chat_example = tokenizer.apply_chat_template(
            [{"role": "user", "content": input_prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=is_reasoning
        )
    return chat_example

MAPPING = {
    "factuality": "Factuality",
    "num_prop": "Number of Propositions",
    "evidence_scope": "Evidence Scope",
    "transformation_level": "Transformation Level"
}

class Drafter:
    """
        Input: context, constraints (context in ["source_text", "passage"])
        Output: state_0
    """
    def __init__(self, model_nickname, tokenizer, target, seed, top_k, top_p, temperature, max_tokens, sampling_n=1, use_exemplars=False):
        self.model_nickname = model_nickname
        self.tokenizer = tokenizer
        self.is_reasoning = False
        self.use_exemplars = use_exemplars
        self.sampling_n = sampling_n
        self.seed = seed
        self.top_k = top_k
        self.top_p = top_p
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        self.exemplar_dict = None
        
        self.target = target
        if target == "passage":
            self.prompt = json.load(open("prompts.json"))["passage_generation"]["drafter"]
        elif target == "option":
            if self.use_exemplars:
                self.prompt_true = json.load(open("prompts.json"))["option_generation"]["drafter_w_ex"]
                self.exemplar_dict = json.load(open("data/exemplars/exemplar_thoughts.json"))
            else:
                self.prompt_true = json.load(open("prompts.json"))["option_generation"]["drafter"]
        else:
            raise ValueError("Target must be either 'passage' or 'option'.")
        
    def set_prompt(self, input_data, constraints):
        if self.target == "passage":
            assert "source_text" in input_data
            assert type(input_data["source_text"]) is str
            input_prompt = self.prompt.replace("{ context }", input_data["source_text"])
            for key, value in constraints.items():
                input_prompt = input_prompt.replace("{ "+ key + " }", str(value))
        
        if self.target == "option":
            assert type(input_data["passage"]) is list
            passage = "\n".join(input_data["passage"])
            
            input_prompt = self.prompt_true.replace("{ passage }", passage) #.replace("{ passage_props }", passage_props)
            input_prompt = input_prompt.replace("{ vocab_level }", constraints["vocab_level"])
            
            constraint_strings = []
            for oidx, c in constraints["options"].items():
                constraint_str = ""
                for key, value in c.items():
                    constraint_str += f"- {MAPPING[key]}: {value}\n"
                    
                constraint_set_str = json.dumps(c)
                constraint_strings.append(constraint_set_str)
                    
                input_prompt = input_prompt.replace("{ option_" + oidx + "_constraints }", constraint_str.strip())
                
            if self.use_exemplars:
                exemplars = []
                for cstr in list(set(constraint_strings)):
                    exemplars.append(self.exemplar_dict[cstr])
                    
                exemplar_str = "\n\n".join([f"[Example {i+1}]\n" + ex for i, ex in enumerate(exemplars)])
                input_prompt = input_prompt.replace("{ exemplars }", exemplar_str)
                
        if "drafter" in LOG_ON or "all" in LOG_ON:
            print("[Input Prompt]")
            print(input_prompt)
            print("---")
        
        chat_example = get_chat_template(self.tokenizer, self.model_nickname, input_prompt, self.is_reasoning)
        return chat_example
        
    def draft(self, llm, input_examples, max_attempt=1):
        examples = input_examples[:]
        
        print("### Drafter ###")
        outputs = dict()
        levels = set()
        for example in examples:
            levels.add(example["level"])
            if self.sampling_n == 1:
                outputs[example["id"]] = {"response": None, "content": None}
            else:
                outputs[example["id"]] = []
                
        all_examples = examples[:]
                
        #### To generate diverse version of passage given the same set of constraints
        for level in sorted(list(levels)):
            examples = [ex for ex in all_examples if ex["level"] == level]
            print(f"### Level {level} - {len(examples)} examples ###")
            for t in range(max_attempt):
                print("Trial:", t+1)
                id_list, input_prompts = [], []
                for example in examples:
                    chat_example = self.set_prompt(example["input_data"], example["constraints"])
                    id_list.append(example["id"])
                    input_prompts.append(chat_example)
                    
                sampling_params = SamplingParams(seed=self.seed+int(level)+t,
                                                top_k=self.top_k,
                                                top_p=self.top_p,
                                                temperature=self.temperature,
                                                max_tokens=self.max_tokens,
                                                n=self.sampling_n)
                    
                response = llm.generate(
                    input_prompts,
                    sampling_params=sampling_params
                )
                for id, res in zip(id_list, response):
                    example_outputs = []
                    if res.outputs:
                        for output in res.outputs:
                            r = output.text
                                
                            json_data = extract_json(r)
                            
                            if "drafter" in LOG_ON or "all" in LOG_ON:
                                print("[Drafter Response]", id)
                                print(r)
                                print("===")
                            
                            if json_data is not None:
                                if self.target == "passage":
                                    try:
                                        passage_sents = sentence_segment(json_data["passage"])
                                        assert len(passage_sents) > 0
                                        if self.sampling_n > 1:
                                            outputs[id].append({
                                                "response": r,
                                                "content": passage_sents
                                            })
                                        else:
                                            outputs[id] = {
                                                "response": r,
                                                "content": passage_sents
                                            }
                                    except:
                                        continue
                                    
                                elif self.target == "option":
                                    try:
                                        assert "stem" in json_data and "options" in json_data and "answer" in json_data
                                        assert type(json_data["stem"]) is str
                                        assert type(json_data["options"]) is dict and len(json_data["options"]) == 4
                                        assert json_data["answer"] in ["A", "B", "C", "D"]
                                        assert all([k in ["A", "B", "C", "D"] for k in json_data["options"].keys()])
                                        
                                        if self.sampling_n > 1:
                                            outputs[id].append({
                                                "response": r,
                                                "content": json_data
                                            })
                                        else:
                                            outputs[id] = {
                                                "response": r,
                                                "content": json_data
                                            }                                   
                                    except:
                                        continue
                                        
                ## Check if all outputs are filled
                ## If not, continue to the next attempt with failed examples
                retry = []
                for example in examples:
                    if example["level"] != level:
                        continue
                    
                    if self.sampling_n > 1:
                        if len(outputs[example["id"]]) == 0:
                            retry.append(example)
                    else:
                        if outputs[example["id"]]["content"] is None:
                            retry.append(example)
                examples = retry
                
                if len(examples) == 0:
                    break
                
            if len(examples) > 0:
                for example in examples:
                    if self.sampling_n == 1:
                        outputs[example["id"]] = {
                            "response": "ERROR - DRAFTING FAILED",
                            "content": None
                        }
                    else:
                        outputs[example["id"]] = [
                            {
                            "response": "ERROR - DRAFTING FAILED",
                            "content": None
                        }]
            
        return outputs

class Planner:
    """
        Input: source_text, current_state, constraints, error_report
        Output: agent_to_call
    """
    def __init__(self, model_nickname, tokenizer, target, seed, top_k, top_p, temperature, max_tokens, args):
        self.model_nickname = model_nickname
        self.tokenizer = tokenizer
        self.is_reasoning = False
        self.target = target
        self.seed = seed
        self.top_k = top_k
        self.top_p = top_p
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        self.off_creativity_enhancement = args.off_creativity_enhancement
        self.off_planner_instruction = args.off_planner_instruction
        self.off_reworder_message = args.off_reworder_message
        
        self.target = target
        if target == "passage":
            if self.off_creativity_enhancement:
                print("Off creativity enhancement in passage planner.")
                self.prompt = json.load(open("prompts.json"))["passage_generation"]["planner_wo_CE"]
            elif self.off_planner_instruction:
                print("Off planner instruction in passage planner.")
                self.prompt = json.load(open("prompts.json"))["passage_generation"]["planner_wo_instruction"]
            else:
                print("Full passage planner.")
                self.prompt = json.load(open("prompts.json"))["passage_generation"]["planner"]
        elif target == "option":
            if self.off_creativity_enhancement:
                print("Off creativity enhancement in option planner.")
                self.prompt_true = json.load(open("prompts.json"))["option_generation"]["planner_wo_CE"]
            elif self.off_planner_instruction:
                print("Off planner instruction in option planner.")
                self.prompt_true = json.load(open("prompts.json"))["option_generation"]["planner_wo_instruction"]
            else:
                print("Full option planner.")
                self.prompt_true = json.load(open("prompts.json"))["option_generation"]["planner"]
        else:
            raise ValueError("Target must be either 'passage' or 'option'.")

    def set_prompt(self, input_data, current_state, constraints, report, history):
        threshold = 2
        if self.target == "passage":
            current_passage = ""
            for idx, sent in enumerate(current_state):
                current_passage += f"({idx+1}) {sent}\n"
            current_passage = current_passage.strip()
        
            input_prompt = self.prompt.replace("{ context }", input_data["source_text"]).replace("{ current_state }", current_passage).replace("{ report }", report).replace("{ history }", history).replace("{ threshold }", str(threshold)).replace("{ threshold+1 }", str(threshold+1))
            for key, value in constraints.items():
                input_prompt = input_prompt.replace("{ "+ key + " }", str(value))
            
        elif self.target == "option":
            passage = "\n".join(input_data["passage"])
            
            input_prompt = self.prompt_true.replace("{ passage }", passage).replace("{ report }", report).replace("{ history }", history).replace("{ threshold }", str(threshold)).replace("{ threshold+1 }", str(threshold+1))
            input_prompt = input_prompt.replace("{ vocab_level }", constraints["vocab_level"])
            for oidx, c in constraints["options"].items():
                constraint_str = ""
                for key, value in c.items():
                    constraint_str += f"- {MAPPING[key]}: {value}\n"
                    
                input_prompt = input_prompt.replace("{ option_" + oidx + "_constraints }", constraint_str.strip())
                
            input_prompt = input_prompt.replace("{ stem }", current_state["stem"])
            for oidx, option in current_state["options"].items():
                input_prompt = input_prompt.replace("{ option_" + oidx + " }", option)
            input_prompt = input_prompt.replace("{ answer }", current_state["answer"])
            
        if "planner" in LOG_ON or "all" in LOG_ON:
            print("[Input Prompt-Planner]")
            print(input_prompt)
            print("---")
            
        chat_example = get_chat_template(self.tokenizer, self.model_nickname, input_prompt, self.is_reasoning)
        return chat_example
    
    def call(self, llm, input_examples, revision_round, max_attempt=1):
        examples = input_examples[:]
        
        print("### Planner ###")
        id2output = dict([(example["id"], None) for example in examples])
        for t in range(max_attempt):
            id_list, input_prompts = [], []
            for example in examples:
                constraints = example["constraints"]
                last_worker = example["trajectory"][revision_round-1]["last_worker"]
                current_state = example["trajectory"][revision_round-1][last_worker]["state"]
                
                history = ""
                if not self.off_planner_instruction:
                    if example["planner_history"]:
                        history = "\n".join(example["planner_history"][-3:])
                    else:
                        history = "This is the first trial."
                
                previous_reports = example["trajectory"][revision_round-1][last_worker]["report"]
                if self.target == "passage":
                    report = "".join([message for message in previous_reports.values() if message != ""])
                elif self.target == "option":
                    report = previous_reports["neutrality"] + "\n"
                    for oidx in ["A", "B", "C", "D"]:
                        report += f"[Option {oidx} Report]\n"
                        report += "".join([message for message in previous_reports["options"][oidx].values() if message != ""])
                        report += "\n"
                if not self.off_reworder_message:
                    ### Add reworder message
                    report += previous_reports.get("reworder_advice", "")
                
                chat_example = self.set_prompt(example["input_data"], current_state, constraints, report, history)
                id_list.append(example["id"])
                input_prompts.append(chat_example)
                
            sampling_params = SamplingParams(seed=self.seed+t,
                                            top_k=self.top_k,
                                            top_p=self.top_p,
                                            temperature=self.temperature,
                                            max_tokens=self.max_tokens,
                                            n=1)
                
            response = llm.generate(
                input_prompts,
                sampling_params=sampling_params
            )
            
            retry_examples = []
            for id, res, example in zip(id_list, response, examples):
                if res.outputs and res.outputs[0].text:
                    r = res.outputs[0].text.strip()
                    
                    if "planner" in LOG_ON or "all" in LOG_ON:
                        print("[Planner Response]")
                        print(r)
                        print("===")
                    
                    if self.is_reasoning:
                        r = r.split("</think>")[-1].strip()
                    
                    if r.strip().startswith("```json"):
                        retry_examples.append(example)
                        continue ## Think First
                    
                    json_data = extract_json(r)
                    
                    if json_data is None:
                        retry_examples.append(example)
                        continue
                    
                    try:
                        action = json_data["action"]
                        assert action in ["Call_Reworder", "Call_Editor"]
                        
                        if not self.off_planner_instruction:
                                assert "message" in json_data
                                assert type(json_data["message"]) is str
                        if self.target == "option":
                            assert "target" in json_data
                            assert type(json_data["target"]) is str and json_data["target"] in ["A", "B", "C", "D"]
                        
                        target = json_data.get("target", None)
                        message = json_data.get("message", "")
                        
                        id2output[id] = {
                            "response": r,
                            "action": action,
                            "target": target,
                            "message": message
                        }
                    except:
                        retry_examples.append(example)
                        
            if len(retry_examples) == 0:
                break
            
            examples = retry_examples
            
        if len(retry_examples) > 0:
            for example in retry_examples:
                if "planner" in example["trajectory"][revision_round-1]:
                    previous_target = example["trajectory"][revision_round-1]["planner"].get("target", None)
                else:
                    random_target = [oidx for oidx, succ in example["success_details"].items() if not succ]
                    if len(random_target) > 0:
                        previous_target = random_target[0]
                    else:
                        previous_target = "A"
                        
                id2output[example["id"]] = {
                    "response": "ERROR - DEFAULT TO EDITOR",
                    "action": "Call_Editor",
                    "target": previous_target,
                    "message": "ERROR - PLANNING FAILED, DEFAULT TO EDITOR"
                }

        return id2output
    
class Editor:
    """
        Input: source_text, current_state, constraints, report
        Output: Thought_i & Action_i
    """
    def __init__(self, model_nickname, tokenizer, target, seed, top_k, top_p, temperature, max_tokens, use_exemplars=False, args=None):
        self.model_nickname = model_nickname
        self.tokenizer = tokenizer
        self.is_reasoning = False
        self.use_exemplars = use_exemplars
        self.target = target
        self.seed = seed
        self.top_k = top_k
        self.top_p = top_p
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.exemplar_dict = None
        
        self.off_planner_instruction = args.off_planner_instruction
        
        self.target = target
        if target == "passage":
            if self.off_planner_instruction:
                self.prompt = json.load(open("prompts.json"))["passage_generation"]["editor_wo_planner_instruction"]
            else:
                self.prompt = json.load(open("prompts.json"))["passage_generation"]["editor"]
        elif target == "option":
            if self.use_exemplars:
                if self.off_planner_instruction:
                    self.prompt_true = json.load(open("prompts.json"))["option_generation"]["editor_w_ex_wo_planner_instruction"]
                else:
                    self.prompt_true = json.load(open("prompts.json"))["option_generation"]["editor_w_ex"]
                self.exemplar_dict = json.load(open("data/exemplars/exemplar_thoughts.json"))
            else:
                if self.off_planner_instruction:
                    self.prompt_true = json.load(open("prompts.json"))["option_generation"]["editor_wo_planner_instruction"]
                else:
                    self.prompt_true = json.load(open("prompts.json"))["option_generation"]["editor"]
        else:
            raise ValueError("Target must be either 'passage' or 'option'.")

    def set_prompt(self, input_data, current_state, constraints, report, target_element_idx=None, target_element=None):
        if self.target == "passage":
            current_passage = ""
            for idx, sent in enumerate(current_state):
                current_passage += f"({idx+1}) {sent}\n"
            current_passage = current_passage.strip()
        
            input_prompt = self.prompt.replace("{ context }", input_data["source_text"]).replace("{ current_state }", current_passage).replace("{ report }", report)
            for key, value in constraints.items():
                input_prompt = input_prompt.replace("{ "+ key + " }", str(value))
                
        elif self.target == "option":
            passage = "\n".join(input_data["passage"])
            
            input_prompt = self.prompt_true.replace("{ passage }", passage).replace("{ report }", report).replace("{ target_element }", target_element) #.replace("{ passage_props }", passage_props)
            input_prompt = input_prompt.replace("{ vocab_level }", constraints["vocab_level"])
            for oidx, c in constraints["options"].items():
                constraint_str = ""
                for key, value in c.items():
                    constraint_str += f"- {MAPPING[key]}: {value}\n"
                    
                input_prompt = input_prompt.replace("{ option_" + oidx + "_constraints }", constraint_str.strip())
                
            if self.use_exemplars:
                target_constrint_set_str = json.dumps(constraints["options"][target_element_idx])
                exemplar = self.exemplar_dict[target_constrint_set_str] if (self.use_exemplars and target_constrint_set_str and target_constrint_set_str in self.exemplar_dict) else None
                if exemplar:
                    input_prompt = input_prompt.replace("{ exemplars }", exemplar)
                
            input_prompt = input_prompt.replace("{ stem }", current_state["stem"])
            for oidx, option in current_state["options"].items():
                input_prompt = input_prompt.replace("{ option_" + oidx + " }", option)
            input_prompt = input_prompt.replace("{ answer }", current_state["answer"])
            
        if "editor" in LOG_ON or "all" in LOG_ON:
            print("[Input Prompt-Editor]")
            print(input_prompt)
            print("---")
            
        chat_example = get_chat_template(self.tokenizer, self.model_nickname, input_prompt, self.is_reasoning)
        return chat_example
    
    def call(self, llm, input_examples, revision_round, max_attempt=1):
        examples = input_examples[:]
        
        print("### Editor ###")
        id2output = dict([(example["id"], None) for example in examples])

        for t in range(max_attempt):
            id_list, input_prompts = [], []
            for example in examples:
                constraints = example["constraints"]
                last_worker = example["trajectory"][revision_round-1]["last_worker"]
                current_state = example["trajectory"][revision_round-1][last_worker]["state"]
                
                report = ""
                if self.off_planner_instruction:
                    ### Error except for vocab_level
                    if self.target == "passage":
                        report = "".join([message for k, message in example["trajectory"][revision_round-1][last_worker]["report"].items() if k != "vocab_level" and message != ""])
                    elif self.target == "option":
                        report = example["trajectory"][revision_round-1][last_worker]["report"]["neutrality"] + "\n"
                        for oidx in ["A", "B", "C", "D"]:
                            report += f"[Option {oidx} Report]\n"
                            report += "".join([message for k, message in example["trajectory"][revision_round-1][last_worker]["report"]["options"][oidx].items() if k != "vocab_level" and message != ""])
                            report += "\n"
                else:
                    ## Planner's message
                    message = example["trajectory"][revision_round]["planner"]["message"]
                    if message != "":
                        report += message
                    
                target_element_idx = None
                target_element = None
                if self.target == "option":
                    target_element_idx = example["trajectory"][revision_round]["planner"]["target"]
                    target_element = f'{target_element_idx}. {current_state["options"][target_element_idx]}'
                
                chat_example = self.set_prompt(example["input_data"], current_state, constraints, report, target_element_idx, target_element)
                id_list.append(example["id"])
                input_prompts.append(chat_example)
                
            sampling_params = SamplingParams(seed=self.seed+t,
                                             top_k=self.top_k,
                                             top_p=self.top_p,
                                             temperature=self.temperature,
                                             max_tokens=self.max_tokens,
                                             n=1)
                
            response = llm.generate(
                input_prompts,
                sampling_params=sampling_params
            )
            
            retry_examples = []
            for id, res, example in zip(id_list, response, examples):
                if res.outputs and res.outputs[0].text:
                    r = res.outputs[0].text.strip()

                    if "editor" in LOG_ON or "all" in LOG_ON:
                        print("[Editor Response]", id)
                        print(r)
                        print("===")
                    
                    if self.is_reasoning:
                        r = r.split("</think>")[-1].strip()
                    
                    if r.strip().startswith("```json"):
                        retry_examples.append(example)
                        continue ## Think First
                    
                    json_data = extract_json(r)
                    
                    if json_data is None:
                        retry_examples.append(example)
                        continue
                    
                    try:
                        if self.target == "passage":
                            sentences = ""
                            for k, sent in json_data.items():
                                assert k.startswith("sentence")
                                sentences += sent.strip() + "\n"
                            sentences = sentences.strip()
                            
                            sent_list = sentence_segment(sentences)
                            assert len(sent_list) > 0
                            
                            id2output[id] = {
                                "response": r,
                                "content": sent_list
                            }
                            
                        elif self.target == "option":
                            assert type(json_data) is dict
                            assert "revised" in json_data
                            assert type(json_data["revised"]) is str
                            
                            id2output[id] = {
                                "response": r,
                                "content": json_data["revised"]
                            }
                        
                    except:
                        retry_examples.append(example)
                            
            if len(retry_examples) == 0:
                break
            examples = retry_examples[:]
            
        if len(retry_examples) > 0:
            for example in retry_examples:
                last_worker = example["trajectory"][revision_round-1]["last_worker"]
                id2output[example["id"]] = {
                    "response": "ERROR - REVISION FAILED, KEEP PREVIOUS STATE",
                    "content": example["trajectory"][revision_round-1][last_worker]["state"]
                }

        return id2output
        
class Reworder:
    """
        Input: Sentence, Dictionary
        Output: Rewording Guide (Alternative expressions or suggesting rewriting)
    """
    def __init__(self, model_nickname, tokenizer, lex, target, seed, top_k, top_p, temperature, max_tokens, args):
        self.model_nickname = model_nickname
        self.tokenizer = tokenizer
        self.lex = lex
        self.is_reasoning = False
        self.target = target
        self.seed = seed
        self.top_k = top_k
        self.top_p = top_p
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        self.off_planner_instruction = args.off_planner_instruction
        
        if self.target == "passage":
            if self.off_planner_instruction:
                self.suggestion_prompt = json.load(open("prompts.json"))["passage_generation"]["reworder"]["suggestion_wo_planner_instruction"]
            else:
                self.suggestion_prompt = json.load(open("prompts.json"))["passage_generation"]["reworder"]["suggestion"]
            self.revision_prompt = json.load(open("prompts.json"))["passage_generation"]["reworder"]["revision"]
        else:
            if self.off_planner_instruction:
                self.suggestion_prompt = json.load(open("prompts.json"))["option_generation"]["reworder"]["suggestion_wo_planner_instruction"]
            else:
                self.suggestion_prompt = json.load(open("prompts.json"))["option_generation"]["reworder"]["suggestion"]
            self.revision_prompt = json.load(open("prompts.json"))["option_generation"]["reworder"]["revision"]
        
    def set_suggestion_prompt(self, input_data, constraints, current_state, report):
        if self.target == "passage":
            current_passage = ""
            for idx, sent in enumerate(current_state):
                current_passage += f"({idx+1}) {sent}\n"
            current_passage = current_passage.strip()
            
            input_prompt = self.suggestion_prompt.replace("{ context }", input_data["source_text"]).replace("{ current_state }", current_passage).replace("{ report }", report)
            for key, value in constraints.items():
                input_prompt = input_prompt.replace("{ "+ key + " }", str(value))
                        
        if self.target == "option":
            passage = "\n".join(input_data["passage"])
            input_prompt = self.suggestion_prompt.replace("{ passage }", passage).replace("{ target_element }", current_state).replace("{ vocab_level }", constraints["vocab_level"]).replace("{ report }", report)
        
        if "reworder" in LOG_ON or "all" in LOG_ON:
            print("[Input Prompt - Reworder's Suggestion]")
            print(input_prompt)
            print("---")
        
        chat_example = get_chat_template(self.tokenizer, self.model_nickname, input_prompt, self.is_reasoning)
        return chat_example
    
    def set_revise_prompt(self, current_state, alternative_dict, constraints):
        constraint_level = constraints["vocab_level"]
        
        if self.target == "passage":
            available_vocab_dict = self.check_avaliable_vocabs(current_state, alternative_dict, constraint_level)
            alternative_list_up = ""
            for sent_idx, vocab_pairs in available_vocab_dict.items():
                sent_list_up = ""
                impossible_targets = []
                for (target, target_level), alternatives in vocab_pairs.items():
                    if target_level == "OOV":
                        continue
                    if len(alternatives) > 0:
                        sent_list_up += f'"{target} ({target_level})" → '
                        for alter, level in alternatives:
                            sent_list_up += f'"{alter} ({level})",'
                        sent_list_up = sent_list_up[:-1]+"\n"
                    else:
                        impossible_targets.append(target)
                if sent_list_up:
                    alternative_list_up += f"Sentence ({sent_idx}):\n" + sent_list_up
                if impossible_targets:
                    alternative_list_up += f'The word(s) "' + '", "'.join(impossible_targets) + f'" in sentence ({sent_idx}) cannot be reworded.\n'  #Replace the phrases containing these words with different information or remove them.

            current_passage = ""
            for idx, sent in enumerate(current_state):
                current_passage += f"({idx+1}) {sent}\n"
            current_passage = current_passage.strip()
            
            input_prompt = self.revision_prompt.replace("{ current_state }", current_passage).replace("{ alternative_list_up }", alternative_list_up).replace("{ vocab_level }", constraint_level)       
        
        if self.target == "option":
            available_vocab_dict = self.check_avaliable_vocabs([current_state], {"1": alternative_dict}, constraint_level)
            alternative_list_up = ""
            vocab_pairs = available_vocab_dict["1"]
            impossible_targets = []
            for (target, target_level), alternatives in vocab_pairs.items():
                if target_level == "OOV":
                    continue
                if len(alternatives) > 0:
                    alternative_list_up += f'{target} ({target_level}) →'
                    for alter, level in alternatives:
                        alternative_list_up += f' {alter} ({level}),'
                    alternative_list_up = alternative_list_up[:-1]+"\n"
                else:
                    impossible_targets.append(target)
                    
            if impossible_targets:
                alternative_list_up += f'The word(s) "' + '", "'.join(impossible_targets) + f'" in the statement cannot be reworded.\n'
            
            input_prompt = self.revision_prompt.replace("{ target_element }", current_state).replace("{ alternative_list_up }", alternative_list_up).replace("{ vocab_level }", constraint_level)
        
        if "reworder" in LOG_ON or "all" in LOG_ON:
            print("[Input Prompt - Reworder's Revision]")
            print(input_prompt)
            print("---")
        
        chat_example = get_chat_template(self.tokenizer, self.model_nickname, input_prompt, self.is_reasoning)
        return chat_example
    
    def suggest(self, llm, input_examples, round, max_attempt=1):
        examples = input_examples[:]
        
        print("### Reworder - Suggestion ###")
        def is_valid_replacement_dict(data, current_state=None):
            if self.target == "passage":
                if not isinstance(data, dict):
                    print("Not a dict")
                    return False

                for sentence_id, replacements in data.items():
                    if not isinstance(sentence_id, str) or not sentence_id.isdigit():
                        print("Sentence ID not valid")
                        return False
                    if not isinstance(replacements, dict):
                        print("Replacements not a dict")
                        return False
                    
                    if int(sentence_id) < 1 or int(sentence_id) > len(current_state):
                        print("Sentence ID out of range")
                        return False  # Ensure the sentence_id is within the range of current_state sentences

                    for word, alt_list in replacements.items():
                        if not isinstance(word, str):
                            print("Word not a string")
                            return False
                        if not isinstance(alt_list, list) or not all(isinstance(alt, str) for alt in alt_list):
                            print("Alt list not valid")
                            return False
                        if word not in current_state[int(sentence_id)-1]:
                            print(f'Word "{word}" not in sentence {sentence_id}')
                            return False  # Ensure the word exists in the current state sentence
                        
            if self.target == "option":
                """
                ```json\n{\n  \"appointed\": [\"named\", \"put in\"],\n  \"resigned\": [\"left\", \"quit\"]\n}\n```
                """
                if not isinstance(data, dict):
                    return False
                
                for word, alt_list in data.items():
                    if not isinstance(word, str):
                        return False
                    if not isinstance(alt_list, list) or not all(isinstance(alt, str) for alt in alt_list):
                        return False
                    
            return True
        
        id2output = dict([(example["id"], None) for example in examples])
        
        for t in range(max_attempt):
            id_list, input_prompts = [], []
            for example in examples:
                last_worker = example["trajectory"][round-1]["last_worker"]
                current_state = example["trajectory"][round-1][last_worker]["state"]
                
                if self.target == "passage":
                    report = ""
                    if self.off_planner_instruction:
                        report = example["trajectory"][round-1][last_worker]["report"]["vocab_level"]
                    else:
                        message = example["trajectory"][round]["planner"]["message"]
                        if message != "":
                            report += message
                    
                    chat_example = self.set_suggestion_prompt(example["input_data"], example["constraints"], current_state, report)
                    
                if self.target == "option":
                    target_element_idx = example["trajectory"][round]["planner"]["target"]
                    target_element = f'{target_element_idx}. {current_state["options"][target_element_idx]}'
                    report = ""
                    if self.off_planner_instruction:
                        report += example["trajectory"][round-1][last_worker]["report"]["options"][target_element_idx]["vocab_level"]
                    else:
                        message = example["trajectory"][round]["planner"]["message"]
                        if message != "":
                            report += message
                    
                    chat_example = self.set_suggestion_prompt(example["input_data"], example["constraints"], target_element, report)
                
                id_list.append(example["id"])
                input_prompts.append(chat_example)
                
            sampling_params = SamplingParams(seed=self.seed+t,
                                             top_k=self.top_k,
                                             top_p=self.top_p,
                                             temperature=self.temperature,
                                             max_tokens=self.max_tokens,
                                             n=1)

            response = llm.generate(
                input_prompts,
                sampling_params=sampling_params
            )
            
            retry_examples = []
            for id, res, example in zip(id_list, response, examples):
                if res.outputs and res.outputs[0].text:
                    r = res.outputs[0].text
                    if r.strip().startswith("```json"):
                        retry_examples.append(example)
                        continue ## Think First
                    
                    if "reworder" in LOG_ON or "all" in LOG_ON:
                        print("[Reworder Suggestion Response]", id)
                        print(r)
                        print("===")
                    
                    json_data = extract_json(r)
                    
                    if json_data is None:
                        print("JSON extraction failed")
                        retry_examples.append(example)
                        continue
                    
                    last_worker = example["trajectory"][round-1]["last_worker"]
                    current_state = example["trajectory"][round-1][last_worker]["state"]
                    if is_valid_replacement_dict(json_data, current_state):
                        id2output[id] = {
                            "response": r,
                            "output": json_data
                        }
                    else:
                        retry_examples.append(example)
            
            ## Check if all outputs are filled
            ## If not, continue to the next attempt with failed examples
            if len(retry_examples) == 0:
                break
            examples = retry_examples[:]
            
        if len(retry_examples) > 0:
            for example in retry_examples:
                id2output[example["id"]] = {
                    "response": "ERROR - REVISION FAILED, KEEP PREVIOUS STATE",
                    "output": {}
                }           
            
        return id2output
    
    def revise(self, llm, input_examples, id2suggestion, round, max_attempt=1):
        examples = input_examples[:]
        
        print("### Reworder - Revision ###")
        
        def is_valid_revised_output(data):
            if not isinstance(data, dict):
                return False

            if "updated" not in data or "message" not in data:
                return False
            
            if not isinstance(data["message"], str):
                return False
            
            if self.target == "passage":
                if not isinstance(data["updated"], list):
                    return False
                if not all(isinstance(sentence, str) for sentence in data["updated"]):
                    return False
                if len(data["updated"]) == 0:
                    return False
                
            if self.target == "option":
                if not isinstance(data["updated"], str):
                    return False
                
            return True
        
        id2output = dict([(example["id"], None) for example in examples])
        
        for t in range(max_attempt):
            id_list, input_prompts = [], []
            for example in examples:
                id = example["id"]
                
                last_worker = example["trajectory"][round-1]["last_worker"]
                
                if self.target == "passage":
                    current_state = example["trajectory"][round-1][last_worker]["state"]
                
                if self.target == "option":
                    target_element_idx = example["trajectory"][round]["planner"]["target"]
                    current_state = example["trajectory"][round-1][last_worker]["state"]["options"][target_element_idx]
                alternative_dict = id2suggestion[id]["alternative_dict"]
                
                chat_example = self.set_revise_prompt(current_state, alternative_dict, example["constraints"])
                
                id_list.append(id)
                input_prompts.append(chat_example)
                
            sampling_params = SamplingParams(seed=self.seed+t,
                                            top_k=self.top_k,
                                            top_p=self.top_p,
                                            temperature=self.temperature,
                                            max_tokens=self.max_tokens,
                                            n=1)
                
            response = llm.generate(
                input_prompts,
                sampling_params=sampling_params
            )
            
            retry_examples = []
            for id, res, example in zip(id_list, response, examples):
                if res.outputs and res.outputs[0].text:
                    r = res.outputs[0].text
                    if r.strip().startswith("```json"):
                        retry_examples.append(example)
                        continue ## Think First
                    
                    json_data = extract_json(r)
                    
                    if "reworder" in LOG_ON or "all" in LOG_ON:
                        print("[Reworder Revision Response]", id)
                        print(json_data)
                        print("===")
                    
                    if json_data is None:
                        retry_examples.append(example)
                        continue
                    
                    if is_valid_revised_output(json_data):
                        id2output[id] = {
                            "response": r,
                            "content": json_data["updated"],
                            "message": json_data["message"]
                        }
                    else:
                        retry_examples.append(example)
            
            ## Check if all outputs are filled
            ## If not, continue to the next attempt with failed examples
            if len(retry_examples) == 0:
                break
            examples = retry_examples[:]
            
        if len(retry_examples) > 0:
            for example in retry_examples:
                if self.target == "passage":
                    previous_state = example["trajectory"][round-1][example["trajectory"][round-1]["last_worker"]]["state"]
                    assert type(previous_state) is list and len(previous_state) > 0, f"Previous state error: {previous_state}"
                if self.target == "option":
                    target_element_idx = example["trajectory"][round]["planner"]["target"]
                    previous_state = example["trajectory"][round-1][example["trajectory"][round-1]["last_worker"]]["state"]["options"][target_element_idx]
                    assert type(previous_state) is str, f"Previous state error: {previous_state}"
                
                print("Revision failed, keep previous state.", previous_state)
                id2output[example["id"]] = {
                    "response": "ERROR - REVISION FAILED, KEEP PREVIOUS STATE",
                    "content": previous_state,
                    "message": "Reworder failed revision, keep previous state."
                }
            
        return id2output
    
    def check_avaliable_vocabs(self, sentences, alternative_dict, constraint_level):
        """
        Return available alternatives
        """
        LEVELS = ["A", "B", "C"]
        available_dict = dict()
        
        for sent_idx, vocab_pairs in alternative_dict.items():
            sent = sentences[int(sent_idx)-1].lower()
            available_dict[sent_idx] = dict()
            for target, alternatives in vocab_pairs.items():
                target = target.lower()
                
                if target not in sent:
                    #print(f"Error - ({sent_idx}) {sent} | {target}\n - PASS")
                    continue
                
                target_levels = self.lex.phrase_level_in_sentence(sent, target)
                if target_levels is None:
                    target_level = "unknown"
                else:
                    target_level = sorted(target_levels)[-1]  # Get the highest level of the target phrase
                
                avails = []
                for alter in alternatives:
                    replaced_sent = sent.replace(target, alter)
                    levels = self.lex.phrase_level_in_sentence(replaced_sent, alter)
                    if levels is None:
                        continue
                    
                    avail = True
                    for l in levels:
                        if LEVELS.index(l) > LEVELS.index(constraint_level):
                            avail = False
                            break
                    if avail:
                        avails.append((alter, sorted(levels)[-1])) ### asign higher level
                available_dict[sent_idx][(target, target_level)] = avails
        return available_dict
    
class Refiner:
    """
        Input: passage
        Output: revised_passage
    """
    def __init__(self, model_nickname, tokenizer, target, seed, top_k, top_p, temperature, max_tokens):
        self.model_nickname = model_nickname
        self.tokenizer = tokenizer
        self.target = target
        self.is_reasoning = False
        self.seed = seed
        self.top_k = top_k
        self.top_p = top_p
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        assert self.target == "passage", f"Refiner target must be 'passage', but got {self.target}"
        
        self.prompt = json.load(open("prompts.json"))["passage_generation"]["refiner"]
            
        
    def set_prompt(self, current_state):
        assert type(current_state) is list
        current_state = "\n".join(current_state)
        input_prompt = self.prompt.replace("{ passage }", current_state)
        
        
        if "refiner" in LOG_ON or "all" in LOG_ON:
            print("[Input Prompt-Refiner]")
            print(input_prompt)
            print("---")
            
        chat_example = get_chat_template(self.tokenizer, self.model_nickname, input_prompt, self.is_reasoning)
        return chat_example
        
    def refine(self, llm, examples, max_attempt=1):
        print("### Refiner ###")
        id2output = dict()
        
        for example in examples:
            id2output[example["id"]] = None
                
        id_list, input_prompts = [], []
        for example in examples:
            last_round = len(example["trajectory"])-1
            last_worker = example["trajectory"][last_round]["last_worker"]
            current_state = example["trajectory"][last_round][last_worker]["state"]
            
            chat_example = self.set_prompt(current_state)
            id_list.append(example["id"])
            input_prompts.append(chat_example)
            
            
        for t in range(max_attempt):
            sampling_params = SamplingParams(seed=self.seed+t,
                                             top_k=self.top_k,
                                             top_p=self.top_p,
                                             temperature=self.temperature,
                                             max_tokens=self.max_tokens,
                                             n=1)
                
            response = llm.generate(
                input_prompts,
                sampling_params=sampling_params
            )
            
            retry_id_list, retry_input_prompts = [], []
            for id, res, input_prompt in zip(id_list, response, input_prompts):
                if res.outputs and res.outputs[0].text:
                    r = res.outputs[0].text
                    if "refiner" in LOG_ON or "all" in LOG_ON:
                        print("[Refiner Response]", id)
                        print(r)
                        print("===")
                    json_data = extract_json(r)
                    
                    if json_data is None:
                        retry_id_list.append(id)
                        retry_input_prompts.append(input_prompt)
                        continue
                    
                    try:
                        id2output[id] = {"response": r,
                                        "passage": sentence_segment(json_data["passage"])}
                    except:
                        retry_id_list.append(id)
                        retry_input_prompts.append(input_prompt)
                            
            ## Check if all outputs are filled
            ## If not, continue to the next attempt with failed examples
            if len(retry_id_list) == 0:
                break
            id_list = retry_id_list[:]
            input_prompts = retry_input_prompts[:]
            
        if len(retry_id_list) > 0:
            for id in retry_id_list:
                example = [ex for ex in examples if ex["id"] == id][0]
                last_round = len(example["trajectory"])-1
                last_worker = example["trajectory"][last_round]["last_worker"]
                previous_state = example["trajectory"][last_round][last_worker]["state"]
                
                id2output[example["id"]] = {
                    "response": "ERROR - REFINEMENT FAILED, KEEP PREVIOUS STATE",
                    "passage": previous_state
                }
                
        return id2output
    