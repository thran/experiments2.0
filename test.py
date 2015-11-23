from utils import data, evaluator
from models.model import AvgModel, ItemAvgModel

data = data.Data("data/matmat/2015-11-20/answers.pd", train_size=0.3)
model = ItemAvgModel()

print(evaluator.Evaluator(data, model))