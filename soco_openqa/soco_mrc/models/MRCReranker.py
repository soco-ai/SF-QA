import torch
from transformers import BertPreTrainedModel
from transformers import BertModel
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss, NLLLoss


class SharedNorm(object):
    def __init__(self):
        self.data = None
        self.log_sm = nn.LogSoftmax(dim=1)
    
    def forward(self, data, group_ids, norm_method):
        if norm_method == 'norm_softmax':
            # do softmax for paras retrieved from same question along batch
            batch_size = data.size(0)
            data = data.view(batch_size//2, -1)
            data = self.log_sm(data)
            data = data.view(batch_size, -1)
        else:
            raise NotImplementedError

        return data

    def __call__(self, logits, group_ids, norm_method):
        return self.forward(logits, group_ids, norm_method)


class BertForQuestionAnsweringWithReranker(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForQuestionAnsweringWithReranker, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)
        self.start_shared_norm = SharedNorm()
        self.end_shared_norm = SharedNorm()
        
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 2)
        self.cls_shared_norm = SharedNorm()
        
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
        group_ids=None,
        norm_method=None,
    ):
        r"""
        start_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`).
            Position outside of the sequence are not taken into account for computing the loss.
        end_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`).
            Position outside of the sequence are not taken into account for computing the loss.

    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`labels` is provided):
            Total span extraction loss is the sum of a Cross-Entropy for the start and end positions.
        start_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length,)`):
            Span-start scores (before SoftMax).
        end_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length,)`):
            Span-end scores (before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.

    Examples::

        from transformers import BertTokenizer, BertForQuestionAnswering
        import torch

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

        question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"
        input_ids = tokenizer.encode(question, text)
        token_type_ids = [0 if i <= input_ids.index(102) else 1 for i in range(len(input_ids))]
        start_scores, end_scores = model(torch.tensor([input_ids]), token_type_ids=torch.tensor([token_type_ids]))

        all_tokens = tokenizer.convert_ids_to_tokens(input_ids)
        answer = ' '.join(all_tokens[torch.argmax(start_scores) : torch.argmax(end_scores)+1])

        assert answer == "a nice puppet"

        """

        sequence_output, pooled_output = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        pooled_output = self.dropout(pooled_output)
        cls_logits = self.classifier(pooled_output)


        outputs = (start_logits, end_logits, cls_logits.clone())
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)
            
            if norm_method:
                start_logits = self.start_shared_norm(start_logits, group_ids, norm_method)
                end_logits = self.end_shared_norm(end_logits, group_ids, norm_method)
                cls_logits = self.cls_shared_norm(cls_logits, group_ids, norm_method)

            loss_fct = NLLLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)

            cls_label = start_positions.clone().clamp_(0, 1)
            cls_loss_fct = NLLLoss(ignore_index=-1)
            cls_loss = cls_loss_fct(cls_logits.clone(), cls_label.clone())

            total_loss = (start_loss + end_loss) / 4 + cls_loss.clone() / 2
            # total_loss = (start_loss + end_loss) / 2
            outputs = (total_loss,) + outputs

        return outputs  # (loss), start_logits, end_logits, (hidden_states), (attentions)
