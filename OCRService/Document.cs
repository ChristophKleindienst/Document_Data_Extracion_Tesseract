using Microsoft.ML.Data;

namespace Document_Data_Extraction_Services
{
    public class Document
    {
        [LoadColumn(0)]
        public string FilePath { get; set; }

        [LoadColumn(1)]
        public string Label { get; set; }

        [LoadColumn(2)]
        public string Text { get; set; }
    }

    public class DocumentPrediction
    {
        [ColumnName("PredictedLabel")]
        public string PredictedLabel { get; set; }
        [ColumnName("Score")]
        public float[] Score { get; set; }
    }
}
