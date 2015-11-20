from utils import data, runner
from models.model import AvgModel

data = data.Data("data/matmat/2015-11-20/answers.pd", train_size=0.3)
model = AvgModel()

runner.Runner(data, model).run()