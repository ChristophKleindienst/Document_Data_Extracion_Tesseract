using Microsoft.ML.Data;

namespace Document_Data_Extraction_Services
{
    public class InvoiceDocument
    {
        [LoadColumn(0)]
        public string FilePath { get; set; }
        [LoadColumn(1)]
        public string Recipient { get; set; }
        [LoadColumn(2)]
        public string Supplier { get; set; }
        [LoadColumn(3)]
        public string InvoiceDate { get; set; }
        [LoadColumn(4)]
        public string InvoiceNumber { get; set; }
        [LoadColumn(5)]
        public string TotalAmount { get; set; }
        [LoadColumn(6)]
        public string RecipientAddress { get; set; }
        [LoadColumn(7)]
        public string Tax { get; set; }
        [LoadColumn(8)]
        public string Text { get; set; }
    }

    public class InvoiceDocumentPrediction
    {
        [ColumnName("PredictedRecipient")]
        public string PredictedRecipient { get; set; }

        [ColumnName("PredictedSupplier")]
        public string PredictedSupplier { get; set; }

        [ColumnName("PredictedInvoiceDate")]
        public string PredictedInvoiceDate { get; set; }

        [ColumnName("PredictedInvoiceNumber")]
        public string PredictedInvoiceNumber { get; set; }

        [ColumnName("PredictedTotalAmount")]
        public string PredictedTotalAmount { get; set; }

        [ColumnName("PredictedRecipientAddress")]
        public string PredictedRecipientAddress { get; set; }

        [ColumnName("PredictedTax")]
        public string PredictedTax { get; set; }
    }
}
