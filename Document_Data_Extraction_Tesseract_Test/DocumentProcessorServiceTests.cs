using Document_Data_Extraction_Services;
using Microsoft.Extensions.Configuration;
using Moq;

namespace Document_Data_Extraction_Tesseract_Test
{
    public class DocumentProcessorServiceTests
    {
        private readonly Mock<IConfiguration> _mockConfiguration;

        public DocumentProcessorServiceTests()
        {
            _mockConfiguration = new Mock<IConfiguration>();
            _mockConfiguration.Setup(config => config["OCRSettings:TessDataPath"]).Returns("..\\..\\..\\..\\Document_Data_Extraction_Tesseract\\tessdata");
            _mockConfiguration.Setup(config => config["OCRSettings:DefaultLanguage"]).Returns("deu");
            _mockConfiguration.Setup(config => config["MLSettings:ModelPath"]).Returns("..\\..\\..\\..\\Document_Data_Extraction_Tesseract\\classifier_data\\documentClassificationModel.zip");
            _mockConfiguration.Setup(config => config["MLSettings:TrainingDataPath"]).Returns("..\\..\\..\\..\\Document_Data_Extraction_Tesseract\\classifier_data\\classifierdata.csv");
        }

        [Fact]
        public void TrainDocumentClassifier_ValidData_TrainsModelSuccessfully()
        {
            // Arrange
            var ocrService = new Mock<IOCRService>();
            ocrService.Setup(service => service.ExtractTextAsync(It.IsAny<string>())).ReturnsAsync("Test");
            var processor = new DocumentProcessorService(ocrService.Object, _mockConfiguration.Object);

            // Act
            processor.TrainDocumentClassifier(Path.Combine(Environment.CurrentDirectory, _mockConfiguration.Object["MLSettings:TrainingDataPath"]), trainNewModel: true);

            // Assert
            Assert.NotNull(processor.Model);
        }
    }
}