using Document_Data_Extraction_Services;
using Microsoft.Extensions.Configuration;
using Serilog;

class Program
{
    public static void ExtractDocumentText(string documentPath, OCRService ocr)
    {
        // executes ocr process and writes the result to the console
        var output = ocr.ExtractText(documentPath);
        Console.WriteLine($"This is the OCR result: {output}");
    }

    public static void ExecuteDocumentClassificationEvaluation(string classificationTrainingDataPath, DocumentProcessorService documentProcessor)
    {

        // executes evaluation of classification prediction process and writes the result to the console
        documentProcessor.TrainDocumentClassifierForEvaluation(classificationTrainingDataPath);
        var classificationMetricResult = documentProcessor.EvaluateDocumentLabelPrediction();
        Console.WriteLine($"Log-loss: {classificationMetricResult?.LogLoss}"); // value between 0 and 1, the lower the value, the better the predictions
        Console.WriteLine($"Per-Class Log-loss: {string.Join(", ", classificationMetricResult?.PerClassLogLoss)}"); // each value provides information about the accuracy of predictions for a specific label

    }

    public static void ExecuteDocumentClassification(string classificationTrainingDataPath, string classificationDocumentPath, DocumentProcessorService documentProcessor, OCRService ocr)
    {
        // executes classification prediction process and writes the result to the console
        documentProcessor.TrainDocumentClassifier(classificationTrainingDataPath);
        var newDocument = new Document { FilePath = classificationDocumentPath };
        newDocument.Text = ocr.ExtractTextAsync(newDocument.FilePath).GetAwaiter().GetResult();
        var classificationResult = documentProcessor.PredictDocumentLabel(newDocument);
        Console.WriteLine($"Predicted Document Type: {classificationResult.PredictedLabel}");
    }

    static void Main(string[] args)
    {
        var configuration = new ConfigurationBuilder()
            .SetBasePath(Directory.GetCurrentDirectory())
            .AddJsonFile(@"appsettings.json", optional: false, reloadOnChange: true)
            .Build();

        Log.Logger = new LoggerConfiguration()
            .WriteTo.File("Logs/log_.log", rollingInterval: RollingInterval.Day)
            .CreateLogger();

        Log.Information("Application started.");
        
        // this provides a path to the csv file which contains learning data for the model to classify documents
        string classificationTrainingDataPath = Path.Combine(Environment.CurrentDirectory, configuration["MLSettings:TrainingDataPath"]);

        var ocr = new OCRService(configuration);
        var documentProcessor = new DocumentProcessorService(ocr, configuration);

        ExecuteDocumentClassificationEvaluation(classificationTrainingDataPath, documentProcessor);

        Log.Information("Application stopped.");
    }
}