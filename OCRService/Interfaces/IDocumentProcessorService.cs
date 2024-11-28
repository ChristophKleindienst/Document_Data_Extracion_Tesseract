using Document_Data_Extraction_Services;
using Microsoft.ML.Data;

public interface IDocumentProcessorService
{
    void TrainDocumentClassifier(string trainingDataPath, bool trainNewModel = true);
    DocumentPrediction PredictDocumentLabel(Document document);
    void TrainDocumentClassifierForEvaluation(string trainingDataPath, bool trainNewModel = true);
    MulticlassClassificationMetrics? EvaluateDocumentLabelPrediction();
}