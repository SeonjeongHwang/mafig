import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch, re, tqdm, nltk
from collections import Counter
from vllm import SamplingParams

def check_passage_constraints(passage, constraints, lex):
    """
    Checks if the passage meets the specified constraints.
    """
    assert isinstance(passage, list) and len(passage) > 0, f"Passage should be a list of sentences. {passage}"
    assert isinstance(constraints, dict), "Constraints should be a dictionary."
    
    report_per_constraints = dict()
    observed_constraints = dict()
    if "passage_length" in constraints:
        report_per_constraints["passage_length"] = ""
        observed_constraints["passage_length"] = False
        pl_category = passage_length_category(passage)
        
        if pl_category == constraints["passage_length"]:
            observed_constraints["passage_length"] = True
            report_per_constraints["passage_length"] += f"The passage has { len(passage) } sentences, which matches the constraint.\n"
        else:
            report_per_constraints["passage_length"] += f"The passage has { len(passage) } sentences, which does not match the constraint.\n"
            
    if "sentence_length" in constraints:
        report_per_constraints["sentence_length"] = ""
        observed_constraints["sentence_length"] = False
        sent2len = dict()
        for idx, sentence in enumerate(passage):
            sent2len[idx+1] = len(sentence.split())
        avg_sentence_length = sum(sent2len.values()) / len(passage)
        sl_category = sentence_length_category(avg_sentence_length)
        
        if sl_category == constraints["sentence_length"]:
            observed_constraints["sentence_length"] = True
            report_per_constraints["sentence_length"] += f"Average sentence length is { round(avg_sentence_length, 1) }, which matches the constraint.\n"
        else:
            report_per_constraints["sentence_length"] += f"Average sentence length is { round(avg_sentence_length, 1) }, which does not match the constraint.\n"
            report_per_constraints["sentence_length"] += f"Sentence lengths: { ', '.join([f'Sentence ({k}): {v} words' for k, v in sent2len.items()]) }\n"
                
    if "vocab_level" in constraints:
        report_per_constraints["vocab_level"] = ""
        observed_constraints["passage_vocab_level"] = True
        observed_constraints["sentence_vocab_level"] = dict()
        proper_level_words = []
        for idx, sentence in enumerate(passage):
            word_levels = lex.word_levels_in_sentence(sentence)
            observed_constraints["sentence_vocab_level"][idx+1] = word_levels
            level_categories = vocab_level_category(word_levels, constraints["vocab_level"])
            
            proper_level_words += level_categories["proper"]
            if len(level_categories["too_hard"]) > 0:
                observed_constraints["passage_vocab_level"] = False
                report_per_constraints["vocab_level"] += f'Sentence ({ idx + 1 }) contains words that are too hard: { level_categories["too_hard"] }\n'
                
        if len(proper_level_words) == 0:
            observed_constraints["passage_vocab_level"] = False
            report_per_constraints["vocab_level"] += f'No words in the passage match the vocabulary level constraint "{constraints["vocab_level"]}".\n'
        
        if observed_constraints["passage_vocab_level"]:
            report_per_constraints["vocab_level"] += f"All words in the passage match the vocabulary level constraint.\n"
                
    if False in observed_constraints.values():
        return False, report_per_constraints, observed_constraints
    else:
        return True, report_per_constraints, observed_constraints
    
def MAE(passage, constraints, lex):
    assert isinstance(passage, list), "Passage should be a list of sentences."
    assert isinstance(constraints, dict), "Constraints should be a dictionary."
    
    mae = dict()
    if "passage_length" in constraints:
        mae["passage_length"] = passage_length_mae(passage, constraints["passage_length"])
    
    if "sentence_length" in constraints:
        avg_sentence_length = sum(len(sentence.split()) for sentence in passage) / len(passage)
        mae["sentence_length"] = sentence_length_mae(avg_sentence_length, constraints["sentence_length"])
    
    if "vocab_level" in constraints:
        word_levels = []
        for sentence in passage:
            word_levels += lex.word_levels_in_sentence(sentence)
        mae["vocab_level"] = vocab_level_mae(word_levels, constraints["vocab_level"])
        
    return mae
        
def passage_length_category(passage):
    """
    Returns the length category of a passage.
    """
    if 5 <= len(passage) <= 10:
        return "short"
    elif 11 <= len(passage) <= 20:
        return "medium"
    elif 21 <= len(passage) <= 30:
        return "long"
    else:
        return "out_of_range"
    
def passage_length_mae(passage, constraint):
    """
    Returns the Mean Absolute Error (MAE) for passage length based on the specified constraint.
    """
    if constraint == "short":
        if 5 <= len(passage) <= 10:
            return 0
        elif len(passage) < 5:
            return len(passage) - 5
        else:
            return len(passage) - 10
    elif constraint == "medium":
        if 11 <= len(passage) <= 20:
            return 0
        elif len(passage) <= 10:
            return len(passage) - 11
        else:
            return len(passage) - 20
    elif constraint == "long":
        if 21 <= len(passage) <= 30:
            return 0
        if len(passage) <= 20:
            return len(passage) - 21
        else:
            return len(passage) - 30
    
def sentence_length_category(avg_sentence_length):
    """
    Returns the length category of a sentence.
    """
    if avg_sentence_length <= 10:
        return "short"
    elif 10 < avg_sentence_length <= 15:
        return "medium"
    elif 15 < avg_sentence_length <= 20:
        return "long"
    else:
        return "out_of_range"
    
def sentence_length_mae(avg_sentence_length, constraint):
    """
    Returns the Mean Absolute Error (MAE) for sentence length based on the specified constraint.
    """
    if constraint == "short":
        if avg_sentence_length <= 10:
            return 0
        else:
            return avg_sentence_length - 10
    elif constraint == "medium":
        if 10 < avg_sentence_length <= 15:
            return 0
        elif avg_sentence_length <= 10:
            return avg_sentence_length - 10.00001
        else:
            return avg_sentence_length - 15
    elif constraint == "long":
        if 15 < avg_sentence_length <= 20:
            return 0
        elif avg_sentence_length <= 15:
            return avg_sentence_length - 15.00001
        else:
            return avg_sentence_length - 20
    
def vocab_level_category(word_levels, level_constraint):
    """
    Returns a dictionary categorizing words based on their CEFR levels.
    """
    LEVELS = ["A", "B", "C"]
    too_hard, proper = [], []
    for word, level in word_levels:
        if level == "OOV":
            continue
        if level == level_constraint:
            proper.append(word)
        elif LEVELS.index(level) > LEVELS.index(level_constraint):
            too_hard.append((word, level))
    return {
        "too_hard": too_hard,
        "proper": proper
    }
    
def vocab_level_mae(word_levels, level_constraint):
    """
    Returns the Mean Absolute Error (MAE) for vocabulary level based on the specified constraint.
    """
    LEVELS = ["A", "B", "C"]
    too_hard = 0
    for word, level in word_levels:
        if level == "OOV":
            continue
        if LEVELS.index(level) > LEVELS.index(level_constraint):
            too_hard += 1
            
    return too_hard

#### OPTION GENERATION
def check_option_constraints(passage, option, constraints, lex, factuality_label, props, es_label, tl_label, report_validity=True):
    """
    Checks if the options meet the specified constraints.
    """
    assert isinstance(option, str), "Option should be a string."
    assert isinstance(constraints, dict), "Constraints should be a dictionary."
    
    report_per_constraints = dict()
    observed_constraints = dict()
    
    observed_constraints["single_sentence"] = True
    if len(nltk.sent_tokenize(option)) > 1:
        observed_constraints["single_sentence"] = False
        report_per_constraints["single_sentence"] = "The statement must be a single sentence.\n"
        
    if "vocab_level" in constraints:
        report_per_constraints["vocab_level"] = ""
        observed_constraints["vocab_level"] = True
        proper_level_words = []
        
        word_levels = lex.word_levels_in_sentence(option)
        observed_constraints["vocab_level_details"] = word_levels
        level_categories = vocab_level_category(word_levels, constraints["vocab_level"])

        if len(level_categories["too_hard"]) > 0:
            observed_constraints["vocab_level"] = False
            report_per_constraints["vocab_level"] += f'The statement contains words that are too hard: { level_categories["too_hard"] }\n'
        else:
            report_per_constraints["vocab_level"] += f"All words in the statement match the vocabulary level constraint.\n"
    
    if "factuality" in constraints:
        observed_constraints["factuality"] = False
        observed_constraints["factuality_details"] = factuality_label
        
        report_per_constraints["factuality"] = f"The statement is labeled as '{ factuality_label }'"
        
        if report_validity:
            if factuality_label == constraints["factuality"]:
                observed_constraints["factuality"] = True
                report_per_constraints["factuality"] += ", which matches the constraint.\n"
            else:
                report_per_constraints["factuality"] += ", which does not match the constraint.\n"
            #return False, report_per_constraints["factuality"], observed_constraints
            
    if "num_prop" in constraints:
        observed_constraints["num_prop"] = False
        observed_constraints["num_prop_details"] = props
        
        report_per_constraints["num_prop"] = f"The statement has { len(props) } proposition(s)"
        
        if report_validity:
            if len(props) == constraints["num_prop"]:
                observed_constraints["num_prop"] = True
                report_per_constraints["num_prop"] += ", which matches the constraint.\n"
            else:
                report_per_constraints["num_prop"] += ", which does not match the constraint.\n"
                
        if len(props) > 1:
            report_per_constraints["num_prop"] += f"\nThe statement can be decomposed into the following propositions: { props }\n"
                
    if "evidence_scope" in constraints and factuality_label != "Not Given":
        observed_constraints["evidence_scope"] = False
        observed_constraints["evidence_scope_details"] = es_label
        
        report_per_constraints["evidence_scope"] = f"Evidence scope is '{ es_label }'"
        
        if report_validity:
            if es_label == constraints["evidence_scope"]:
                observed_constraints["evidence_scope"] = True
                report_per_constraints["evidence_scope"] += ", which matches the constraint.\n"
            else:
                report_per_constraints["evidence_scope"] += ", which does not match the constraint.\n"
            
    if "transformation_level" in constraints and factuality_label != "Not Given":
        report_per_constraints["transformation_level"] = ""
        observed_constraints["transformation_level"] = False
        observed_constraints["transformation_level_details"] = tl_label
        
        report_per_constraints["transformation_level"] += f"Transformation level is '{ tl_label }'"
        
        if report_validity:
            if tl_label == constraints["transformation_level"]:
                observed_constraints["transformation_level"] = True
                report_per_constraints["transformation_level"] += ", which matches the constraint.\n"
            else:
                report_per_constraints["transformation_level"] += ", which does not match the constraint.\n"
                
    if False in observed_constraints.values():
        return False, report_per_constraints, observed_constraints
    else:
        return True, report_per_constraints, observed_constraints
    
import json
import nltk
from nltk import pos_tag, word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

class Lexi_nltk:

    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.dictionary = json.load(open("Dictionary_nltk.json"))
        self.level2num = {"A": 1, "B": 2, "C": 3}
        self.CEFR2threelevels = {"A1": "A", "A2": "A", "B1": "B", "B2": "B", "C1": "C"}
        self.stoppos = ("NNP", "NNPS", "CD", "SYM", ".", ",")

    def nltk_pos_to_wordnet(self, pos_tag_str: str):
        if pos_tag_str.startswith("J"):
            return wordnet.ADJ
        if pos_tag_str.startswith("V"):
            return wordnet.VERB
        if pos_tag_str.startswith("N"):
            return wordnet.NOUN
        if pos_tag_str.startswith("R"):
            return wordnet.ADV
        return wordnet.NOUN  # fallback

    def normalize_pos_for_dict(self, pos_tag_str: str, token: str) -> str:
        if pos_tag_str.startswith("V"):
            return "VB"
        if pos_tag_str.startswith("N"):
            return "NN"
        if pos_tag_str.startswith("J"):
            return "JJ"
        if pos_tag_str.startswith("R"):
            return "RB"
        if pos_tag_str in ("PRP", "PRP$", "WP", "WP$"):
            return "PRP"
        if pos_tag_str in ("DT", "WDT"):
            return "DT"
        if pos_tag_str == "MD":
            return "MD"
        if pos_tag_str == "RP":
            return "RP"
        if pos_tag_str == "CD":
            return "CD"
        if pos_tag_str == "UH":
            return "UH"
        if pos_tag_str == "CC":
            return "CC"
        if pos_tag_str == "SYM":
            return "SYM"
        if pos_tag_str == "IN":
            return "IN"
        return "NN"

    def word_level(self, lemma: str, dict_pos: str) -> str:
        lemma = lemma.lower()
        if lemma in self.dictionary and dict_pos in self.dictionary[lemma]:
            cefr = self.dictionary[lemma][dict_pos]
            return self.CEFR2threelevels.get(cefr, "OOV")
        return "OOV"

    def word_levels_in_sentence(self, sentence: str):
        tokens = word_tokenize(sentence)
        pos_tags = pos_tag(tokens)

        word_levels = []
        for token, pos in pos_tags:
                
            if pos in self.stoppos:
                continue

            wn_pos = self.nltk_pos_to_wordnet(pos)
            lemma = self.lemmatizer.lemmatize(token.lower(), wn_pos)
            dict_pos = self.normalize_pos_for_dict(pos, token)

            level = self.word_level(lemma, dict_pos)
            if level != "OOV":
                word_levels.append((token, level))
        return word_levels

    def phrase_level_in_sentence(self, sentence: str, phrase: str):
        sent_tokens = word_tokenize(sentence)
        phrase_tokens = word_tokenize(phrase)
        if not phrase_tokens:
            return None

        start_idx = -1
        for i in range(len(sent_tokens) - len(phrase_tokens) + 1):
            if sent_tokens[i:i + len(phrase_tokens)] == phrase_tokens:
                start_idx = i
                break
        if start_idx == -1:
            return None

        span_tokens = sent_tokens[start_idx:start_idx + len(phrase_tokens)]
        pos_tags = pos_tag(span_tokens)

        levels = []
        for token, pos in pos_tags:
            if pos in self.stoppos:
                continue
            wn_pos = self.nltk_pos_to_wordnet(pos)
            lemma = self.lemmatizer.lemmatize(token.lower(), wn_pos)
            dict_pos = self.normalize_pos_for_dict(pos, token)

            level = self.word_level(lemma, dict_pos)
            if level != "OOV":
                levels.append(level)

        return levels if levels else None
    
class Propositionalizer:
    def __init__(self, seed):
        self.start_marker = '<s>'
        self.end_marker = '</s>'
        self.separator = '\n'
        
        self.sampling_params = SamplingParams(seed=seed,
                                              max_tokens=4096)
                        
    def get_propositions(self, llm, tokenizer, id_list, sentence_sets, is_passage=False):
        if is_passage:
            assert type(sentence_sets[0]) == list, "When is_passage is True, sents should be a list of list of sentences."
            id2sentences = dict()
            for id, passage_sents in zip(id_list, sentence_sets):
                id2sentences[id] = [[sent] for sent in passage_sents]
        else:
            assert type(sentence_sets[0]) == str, "When is_passage is False, sents should be a list of sentences."
            id2sentence = dict()
            for id, sent in zip(id_list, sentence_sets):
                id2sentence[id] = [sent]
        
        def create_propositions_input(input_sents: list) -> str:
            propositions_input = ''
            for sent in input_sents:
                propositions_input += f'{self.start_marker} ' + sent + f' {self.end_marker}{self.separator}'
            propositions_input = propositions_input.strip(f'{self.separator}')
            if len(tokenizer.encode(propositions_input)) > 4096:
                propositions_input = tokenizer.decode(tokenizer.encode(propositions_input)[:4096]) + f' {self.end_marker}{self.separator}'

            messages = [
                {'role': 'user', 'content': propositions_input}
            ]
            example = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            return example

        def process_propositions_output(text):
            pattern = re.compile(f'{re.escape(self.start_marker)}(.*?){re.escape(self.end_marker)}', re.DOTALL)
            output_grouped_strs = re.findall(pattern, text)
            predicted_grouped_propositions = []
            for grouped_str in output_grouped_strs:
                grouped_str = grouped_str.strip(self.separator)
                props = [x[2:] for x in grouped_str.split(self.separator)]
                predicted_grouped_propositions.append(list(set(props)))
            return predicted_grouped_propositions
        
        id2result = dict([(id, []) for id in id_list])
        for _ in range(5):
            if is_passage:
                input_prompts = [create_propositions_input(sents) for sents in sentence_sets] ## sent: list of sentences
            else:
                input_prompts = [create_propositions_input([sent]) for sent in sentence_sets]
                
            responses = llm.generate(
                input_prompts,
                sampling_params=self.sampling_params
            )
            
            retry_id_list, retry_sentence_sets = [], []
            for id, response, sent_set in zip(id_list, responses, sentence_sets):
                result = process_propositions_output(response.outputs[0].text)
                if is_passage:
                    if len(result) == 0:
                        retry_id_list.append(id)
                        retry_sentence_sets.append(sent_set)
                    else:
                        id2result[id] = result
                else:    
                    if len(result) == 0:
                        retry_id_list.append(id)
                        retry_sentence_sets.append(sent_set)
                    else:
                        id2result[id] = result[0]
                        
            if len(retry_id_list) == 0:
                break
            else:
                id_list = retry_id_list
                sentence_sets = retry_sentence_sets
                
        if len(retry_id_list) > 0:
            for id in retry_id_list:
                if is_passage:
                    id2result[id] = id2sentences[id]
                else:
                    id2result[id] = id2sentences[id]
        
        return id2result
    
def extract_label(response):
    """
    {"label": "..."}
    """
    match = re.search(r"```json\s*({.*?})\s*```", response, re.DOTALL)
    if match:
        json_str = match.group(1)
        try:
            data = json.loads(json_str)
            label = data.get("label", None)
            return label
        except json.JSONDecodeError:
            return None
    return None

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

class OptionNeutralityEvaluator:
    def __init__(self, model_nickname, n=5):
        self.model_nickname = model_nickname
        if n == 1:
            self.sampling_params = SamplingParams(seed=42,
                                                  max_tokens=2000,
                                                  temperature=0,
                                                  n=1)
        else:
            self.sampling_params = SamplingParams(seed=42,
                                                  max_tokens=2000,
                                                  n=n)
        
        self.template = json.load(open("prompts.json"))["evaluator"]["neutrality"]
    
    def evaluate(self, llm, tokenizer, id_list, passages, options):
        print("Evalating Option Neutrality...")
        
        examples = []
        for passage, option_set in zip(passages, options):
            input_prompt = self.template.replace("{ passage }", " ".join(passage))
            for oidx, option in option_set.items():
                input_prompt = input_prompt.replace("{ option_" + str(oidx) + " }", option)
            
            chat_example = get_chat_template(tokenizer, self.model_nickname, input_prompt, False)
            examples.append(chat_example)
        
        id2results = dict([(id, []) for id in id_list])
        responses = llm.generate(
            examples,
            sampling_params=self.sampling_params
        )
            
        for id, example, response in zip(id_list, examples, responses):
            for output in response.outputs:
                json_data = extract_json(output.text)
                if json_data is None:
                    print("Warning: JSON parsing failed.")
                    continue
                if "result" in json_data and json_data["result"] in ["acceptable", "unacceptable"] and "reason" in json_data:
                    id2results[id].append(json_data)
                else:
                    print("Warning: Invalid result value.")
                    continue
                    
        id2result = dict()
        for id, preds in id2results.items():
            if len(preds) == 0:
                id2result[id] = {"result": "unacceptable", "reason": "ERROR - Failed to determine neutrality."}
                continue
            
            predictions = [x["result"] for x in preds]
            if "unacceptable" in predictions:
                message = [x["reason"] for x in preds if x["result"] == "unacceptable"][0]
                id2result[id] = {"result": "unacceptable", "reason": message}
            else:
                id2result[id] = {"result": "acceptable", "reason": ""}
            
        return id2result
    
class FactualityEvaluator:
    def __init__(self, model_nickname, n=5):
        self.model_nickname = model_nickname
        if n == 1:
            self.sampling_params = SamplingParams(seed=42,
                                                  max_tokens=2000,
                                                  temperature=0,
                                                  n=1)
        else:
            self.sampling_params = SamplingParams(seed=42,
                                                  max_tokens=2000,
                                                  n=n)
        
        self.template = json.load(open("prompts.json"))["evaluator"]["factuality"]
    
    def evaluate(self, llm, tokenizer, id_list, passages, statements):
        print("Evalating Factuality...")
        
        examples = []
        for passage, statement in zip(passages, statements):
            input_prompt = self.template.replace("{ passage }", " ".join(passage)).replace("{ statement }", statement)
            
            chat_example = get_chat_template(tokenizer, self.model_nickname, input_prompt, False)
            examples.append(chat_example)
            
        id2results = dict([(id, []) for id in id_list])
        responses = llm.generate(
            examples,
            sampling_params=self.sampling_params
        )
            
        for id, example, response in zip(id_list, examples, responses):
            for output in response.outputs:
                pred = extract_label(output.text)
                if pred is None:
                    continue
                if pred in ["True", "False", "Not Given"]:
                    id2results[id].append(pred)
                    
        id2result = dict()
        for id, preds in id2results.items():
            if len(preds) == 0:
                id2result[id] = "True"
            else:
                major, cnt = Counter(preds).most_common(1)[0]
                id2result[id] = major
            
        return id2result
    
class ComplexityEvaluator:
    def __init__(self, model_nickname, n=5):
        self.model_nickname = model_nickname
        if n == 1:
            self.sampling_params = SamplingParams(seed=42,
                                                  max_tokens=2000,
                                                  temperature=0,
                                                  n=1)
        else:
            self.sampling_params = SamplingParams(seed=42,
                                                  max_tokens=2000,
                                                  n=n)
        
        self.es_template = json.load(open("prompts.json"))["evaluator"]["es"]
        self.tl_true_template = json.load(open("prompts.json"))["evaluator"]["tl_true"]
        self.tl_false_template = json.load(open("prompts.json"))["evaluator"]["tl_false"]  
    
    def evaluate_es(self, llm, tokenizer, id_list, passages, statements, factualities):
        print("Evalating Evidence Scope...")
        
        examples = []
        for passage, statement, factuality in zip(passages, statements, factualities):
            enu_passage = ""
            for sid, sent in enumerate(passage):
                enu_passage += f"({ sid + 1 }) { sent }\n"
            enu_passage = enu_passage.strip()
            
            input_prompt = self.es_template.replace("{ passage }", enu_passage).replace("{ statement }", statement).replace("{ factuality }", factuality)
            chat_example = get_chat_template(tokenizer, self.model_nickname, input_prompt, False)
            examples.append(chat_example)
            
        id2results = dict([(id, []) for id in id_list])
        responses = llm.generate(
            examples,
            sampling_params=self.sampling_params
        )
        
        for id, example, response in zip(id_list, examples, responses):
            for output in response.outputs:
                #print(output.text)
                pred = extract_label(output.text)
                if pred is None:
                    continue
                if pred in ["Single", "Inter"]:
                    id2results[id].append(pred)
                    
        id2result = dict()
        for id, preds in id2results.items():
            if len(preds) == 0:
                id2result[id] = "Single"
            else:
                major, cnt = Counter(preds).most_common(1)[0]
                id2result[id] = major
            
        return id2result
                
    def evaluate_tl(self, llm, tokenizer, id_list, passages, statements, factualities):
        print("Evalating Transformation Level...")
        
        examples = []
        for passage, statement, factuality in zip(passages, statements, factualities):
            enu_passage = ""
            for sid, sent in enumerate(passage):
                enu_passage += f"({ sid + 1 }) { sent }\n"
            enu_passage = enu_passage.strip()
            
            if factuality == "True":
                input_prompt = self.tl_true_template.replace("{ passage }", enu_passage).replace("{ statement }", statement)
            else:
                input_prompt = self.tl_false_template.replace("{ passage }", enu_passage).replace("{ statement }", statement)
            chat_example = get_chat_template(tokenizer, self.model_nickname, input_prompt, False)
            examples.append(chat_example)
            
        id2results = dict([(id, []) for id in id_list])
        responses = llm.generate(
            examples,
            sampling_params=self.sampling_params
        )
            
        for id, example, response in zip(id_list, examples, responses):
            for output in response.outputs:
                pred = extract_label(output.text)
                if pred is None:
                    continue
                if pred in ["Word Matching", "Paraphrasing", "Inference"]:
                    id2results[id].append(pred)
                    
        id2result = dict()
        for id, preds in id2results.items():
            if len(preds) == 0:
                id2result[id] = "Word Matching"
            else:
                major, cnt = Counter(preds).most_common(1)[0]
                id2result[id] = major
            
        return id2result