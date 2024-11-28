using Document_Data_Extraction_Services;
using Microsoft.Extensions.Configuration;
using Moq;

namespace Document_Data_Extraction_Tesseract_Test
{
    public class OCRServiceTests
    {
        private readonly Mock<IConfiguration> _mockConfiguration;

        public OCRServiceTests()
        {
            _mockConfiguration = new Mock<IConfiguration>();
            _mockConfiguration.Setup(config => config["OCRSettings:TessDataPath"]).Returns("..\\..\\..\\..\\Document_Data_Extraction_Tesseract\\tessdata");
            _mockConfiguration.Setup(config => config["OCRSettings:DefaultLanguage"]).Returns("deu");
        }

        [Fact]
        public void ExtractText_ReturnsExpectedText()
        {
            // Arrange
            var ocrService = new OCRService(_mockConfiguration.Object);
            var testImagePath = Path.Combine(Environment.CurrentDirectory, @"..\..\..\ExtractText_ReturnsExpectedText.png");

            // Act
            var result = ocrService.ExtractText(testImagePath);

            // Assert
            Assert.False(string.IsNullOrEmpty(result));
            Assert.Contains("das ist ein test", result);
        }

        [Fact]
        public async Task ExtractTextAsync_ReturnsExpectedText()
        {
            // Arrange
            var ocrService = new OCRService(_mockConfiguration.Object);
            var testImagePath = Path.Combine(Environment.CurrentDirectory, @"..\..\..\ExtractText_ReturnsExpectedText.png");

            // Act
            var result = await ocrService.ExtractTextAsync(testImagePath);

            // Assert
            Assert.False(string.IsNullOrEmpty(result));
            Assert.Contains("das ist ein test", result);
        }
    }
}