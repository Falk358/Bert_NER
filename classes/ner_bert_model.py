from transformers import BertForTokenClassification

from torch import nn



class NerBertModel(nn.Module):

    def __init__(self, unique_labels): 

        super(NerBertModel, self).__init__()

        self.model = BertTokenClassification.from_pretrained("bert-base-cased", num_labels = len(unique_labels))



    def forward(self, input_id, mask, label):

        output = self.model(input_ids = input_id, attention_mask = mask, labels = label, return_dict = False)

        return output
