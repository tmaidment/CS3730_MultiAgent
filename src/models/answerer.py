import torch
from .resnet import Resnet18FeatureExtractor

class Answerer(torch.nn.Module):

    def __init__(self, args):

        super().__init__()

        if args.image_feature_extractor == 'resnet-18':
            self.image_feature_extractor = Resnet18FeatureExtractor()
        else:
            raise NotImplementedError(f'{args.image_feature_extractor}'
                ' not an implemented image feature extractor.')
                
        self.category_embedding = torch.nn.Embedding(
            args.num_categories,
            args.category_embedding_size)
        self.word_embedding = torch.nn.Embedding(
            args.word_vocab_size, 
            args.answerer_word_embedding_size)
        self.qustion_lstm = torch.nn.LSTM(
            args.answerer_word_embedding_size,
            args.answerer_lstm_hidden_size)
        
        num_answerer_features = args.answerer_lstm_size \
            + args.category_embedding_size \
            + args.image_features_size \
            + 8
        
        self.response_mlp = torch.nn.Sequential(
            torch.nn.Linear(
                num_answerer_features,
                args.answerer_response_hidden_size),
            torch.nn.Dropout(),
            torch.nn.ReLU(),
            torch.nn.Linear(
                args.answerer_response_hidden_size,
                args.answerer_response_hidden_size),
            torch.nn.Dropout(),
            torch.nn.ReLU(),
            torch.nn.Linear(
                args.answerer_response_hidden_size,
                2)
        )
    
    def forward(self, question, image, category, spatial,
        question_init_state=None):
        embedded_question = self.word_embedding(question)
        _, question_state = self.qustion_lstm(
            embedded_question, 
            question_init_state)
        question_rep = question_state[0][-1]
        image_rep = self.image_feature_extractor(image)
        category_rep = self.category_embedding(category)
        answerer_features = torch.cat((question_rep, image_rep, 
            category_rep, spatial), dim=1)
        response_scores = self.response_mlp(answerer_features)
        return response_scores, question_state

        
