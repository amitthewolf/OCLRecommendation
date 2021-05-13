
from sklearn.metrics import roc_auc_score

class OperatorClassifier:



    def ScoreOperator(self,train_preds, test_preds, y_train, y_test):
        RocAucScores = []
        for index in range(len(y_train)):
            try:
                RocAucScores.append(roc_auc_score(y_train[index], train_preds[index]))
            except:
                pass
        SumScores = sum(RocAucScores)
        TrainFinalScore = SumScores / len(RocAucScores)
        print(" Train Roc Auc Score - :", TrainFinalScore)
        RocAucScores = []
        for index in range(len(y_test)):
            try:
                RocAucScores.append(roc_auc_score(y_test[index], test_preds[index]))
            except:
                pass
        SumScores = sum(RocAucScores)
        TestFinalScore = SumScores / len(RocAucScores)
        print(" Test Roc Auc Score - :", TestFinalScore)
        return TrainFinalScore, TestFinalScore