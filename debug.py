import os
from transformers import AutoTokenizer
from typing import Sequence


def stride_chunks(l: Sequence, win_len: int, stride_len: int):
    s_id = 0
    e_id = min(len(l), win_len)

    while True:
        yield s_id, l[s_id:e_id]

        if e_id == len(l):
            break

        s_id = s_id + stride_len
        e_id = min(s_id + win_len, len(l))


tokenizer = AutoTokenizer.from_pretrained('resources/squad-chunk-global-norm-2016bm25-bert-large-reranker',use_fast=True)

doc='Srđa Popović (lawyer) Responding to such criticism Popović said in early 1994: "Well, I\'m a lawyer, so technically yes, I am committing an act of treason under Serbian laws. But I distinguish between the interests of the Serbian state and the Serbian people and I think these interests are opposed at this moment. The military defeat of the Milošević government is in the best interest of the Serbian people. It is something that every good Serbian patriot should wish for. I don\'t think I betrayed my people". Pressed further to clarify his position on the ramifications of such an act such as an implication that a violent foreign intervention can solve certain political problems, like the problem of an armed Serb secession from Bosnia-Herzegovina, or from Croatia, as well as his position on the inevitable civilian death toll if such an action is to occur, Popović said: "I think that\'s an unfair question. If I see somebody trying to murder somebody else, of course my duty is to try to stop him. I\'m not saying that by doing so and applying violence to the situation, I\'m actually trying to help those people lead a good life. I don\'t know what they will do once they leave the scene. What I see Serbs doing in Bosnia is committing an act of aggression against a state that has been recognized by United Nations, and I see them committing genocide. I think that both of these things should be stopped. Of course, stopping it would not solve the problem of how these people will live next to each other in the future, but first you have to stop the crimes. The international community has an obligation to do so, under the Genocide Convention and the United Nations Charter. They have an obligation to use force to stop aggression, and to stop genocide. In any armed conflict there will be civilian casualties. Unfortunately, that\'s something that can\'t be avoided. But I don\'t think that this fact should prevent the international community from doing what they are obliged to do under the international law: stopping the aggression, stopping the genocide. It sounds nice to advocate peaceful means, but it is not realistic. I return to this parallel: If you see some big guy beating a kid in the street, it would be very good if you could go to him and say \'Please stop this, you shouldn\'t be doing this, it is uncivilized. This poor guy cannot defend himself.\' No, if that doesn\'t work, you call the police, who have to use violence. At this point in history you have to revert to violence to stop crime." Asked whether he signed the particular petition because of the circumstances rather than principally thinking that such measures solve problems, Popović said: "I\'ll go even further. I signed this document knowing perfectly well that this will never happen. I did it as a gesture to show that I realized who\'s the main culprit in the Yugoslav conflict. And I wanted to express my opinion that this government would deserve it, even though it will never happen".'

doc_enc = tokenizer.encode_plus(doc, add_special_tokens=False, return_offsets_mapping=True, truncation=False)

q = 'What did Luther think was required to stop the violence?'

q_toks = tokenizer.tokenize(q)
q_enc = tokenizer.encode_plus(q, return_offsets_mapping=True)
seq_pair_added_toks = tokenizer.model_max_length - tokenizer.max_len_sentences_pair
window_len = 512 - len(q_toks) - seq_pair_added_toks
stride = 128

for base_idx, chunk_mapping in stride_chunks(doc_enc['offset_mapping'], window_len, stride):
    import pdb; pdb.set_trace()
    chunk_st = chunk_mapping[0][0]
    chunk_ed = chunk_mapping[-1][1]
    chunk = doc[chunk_st: chunk_ed]
    temp = tokenizer.encode_plus(q, chunk, return_offsets_mapping=True, truncation=False)

    new_dict = {}
    # add last [SEP]
    chunk_input_ids = doc_enc['input_ids'][base_idx:base_idx+len(chunk_mapping)]+[102]
    chunk_token_type_ids = [1]*len(chunk_mapping) + [1]
    chunk_attention_mask = [1]*len(chunk_mapping) + [1]
    tmp_chunk_offset_mapping = doc_enc['offset_mapping'][base_idx:base_idx+len(chunk_mapping)]
    base_offset =  tmp_chunk_offset_mapping[0][0]
    chunk_offset_mapping = []
    for offset in tmp_chunk_offset_mapping:
        chunk_offset_mapping.append((offset[0]-base_offset, offset[1]-base_offset))
    import pdb; pdb.set_trace()
    chunk_offset_mapping.append((0, 0))

    new_dict['input_ids'] = q_enc['input_ids'] + chunk_input_ids
    new_dict['token_type_ids'] = q_enc['token_type_ids'] + chunk_token_type_ids
    new_dict['attention_mask'] = q_enc['attention_mask'] + chunk_attention_mask
    new_dict['offset_mapping'] = q_enc['offset_mapping'] + chunk_offset_mapping



