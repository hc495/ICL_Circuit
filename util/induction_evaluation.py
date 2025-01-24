def get_induction_correctness_for_single_layer(ICL_attention, experimentor, sample_index, layer):
    demo_indexs = experimentor.demonstration_sampler[sample_index]
    demo_labels = []
    for demo_index in demo_indexs:
        demo_labels.append(experimentor.get_label_space().index(experimentor.demonstration_set()[demo_index][1]))
    query_label = experimentor.get_label_space().index(experimentor.test_set()[sample_index][1])
    
    res = []
    for heads in ICL_attention[sample_index][layer]:
        temp = 0
        for i in range(len(demo_labels)):
            if demo_labels[i] == query_label:
                temp += heads[2 * i + 1]
            else:
                temp -= heads[2 * i + 1]
        res.append(temp.item())
    return res

def get_induction_magnitude_for_single_layer(ICL_attention, experimentor, sample_index, layer):
    demo_indexs = experimentor.demonstration_sampler[sample_index]
    demo_labels = []
    for demo_index in demo_indexs:
        demo_labels.append(experimentor.get_label_space().index(experimentor.demonstration_set()[demo_index][1]))
    query_label = experimentor.get_label_space().index(experimentor.test_set()[sample_index][1])
    
    res = []
    for heads in ICL_attention[sample_index][layer]:
        temp = 0
        for i in range(len(demo_labels)):
            temp += heads[2 * i + 1]
        res.append(temp.item())
    return res

def tokenized_length(tokenizer, prompt):
    tkized = tokenizer(prompt)['input_ids']
    return len(tkized)

def get_theresold_magnitude_from_prompt(tokenizer, prompt, induction_threthold_times, k):
    tkized = tokenizer(prompt)['input_ids']
    return induction_threthold_times * k / len(tkized)

def get_theresold_correctness_from_prompt(tokenizer, prompt, induction_threthold_times, label_space_length, k):
    return get_theresold_magnitude_from_prompt(tokenizer, prompt, induction_threthold_times, k) / label_space_length

def normalize(vector):
    _sum = sum(vector)
    return [x/_sum for x in vector]