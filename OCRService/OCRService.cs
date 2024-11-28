using Microsoft.Extensions.Configuration;
using Serilog;
using Tesseract;

namespace Document_Data_Extraction_Services
{
    public class OCRService : IOCRService
    {
        private readonly string _tessDataPath;
        private readonly string _language;

        public OCRService(IConfiguration configuration)
        {
            _tessDataPath = Path.Combine(Environment.CurrentDirectory, configuration["OCRSettings:TessDataPath"]);
            _language = configuration["OCRSettings:DefaultLanguage"];
        }

        public string ExtractText(string imagePath)
        {
            try
            {
                using var engine = new TesseractEngine(_tessDataPath, _language, EngineMode.Default);
                using var img = Pix.LoadFromFile(imagePath);
                using var page = engine.Process(img);

                return page.GetText();
            }
            catch (FileNotFoundException ex)
            {
                Log.Error(ex, "File not found: {FilePath}", imagePath);
                return string.Empty;
            }
            catch (TesseractException ex)
            {
                Log.Error(ex, "OCR processing error for file: {FilePath}", imagePath);
                return string.Empty;
            }
            catch (Exception ex)
            {
                Log.Fatal(ex, "Unexpected error during OCR processing for file: {FilePath}", imagePath);
                return string.Empty;
            }
        }

        public async Task<string> ExtractTextAsync(string imagePath)
        {
            try
            {
                using var engine = new TesseractEngine(_tessDataPath, _language, EngineMode.Default);
                using var img = await Task.Run(() => Pix.LoadFromFile(imagePath));
                using var page = await Task.Run(() => engine.Process(img));

                return page.GetText();
            }
            catch (FileNotFoundException ex)
            {
                Log.Error(ex, "File not found: {FilePath}", imagePath);
                return string.Empty;
            }
            catch (TesseractException ex)
            {
                Log.Error(ex, "OCR processing error for file: {FilePath}", imagePath);
                return string.Empty;
            }
            catch (Exception ex)
            {
                Log.Fatal(ex, "Unexpected error during OCR processing for file: {FilePath}", imagePath);
                return string.Empty;
            }
        }

        public void GenerateBoundingBoxesDataFile(string imagePath)
        {
            var boundingBoxes = new List<WordBox>();

            using var engine = new TesseractEngine(_tessDataPath, _language, EngineMode.Default);
            using var img = Pix.LoadFromFile(imagePath);
            using var page = engine.Process(img);

            var iterator = page.GetIterator();
            iterator.Begin();
            do
            {
                var word = iterator.GetText(PageIteratorLevel.Word);
                if (!string.IsNullOrEmpty(word))
                {
                    var box = iterator.TryGetBoundingBox(PageIteratorLevel.Word, out var boundingBox);
                    if (box)
                    {
                        boundingBoxes.Add(new WordBox
                        {
                            Word = word,
                            X1 = boundingBox.X1,
                            Y1 = boundingBox.Y1,
                            X2 = boundingBox.X2,
                            Y2 = boundingBox.Y2
                        });
                    }
                }
            } while (iterator.Next(PageIteratorLevel.Word));

            var jsonOutput = Newtonsoft.Json.JsonConvert.SerializeObject(boundingBoxes, Newtonsoft.Json.Formatting.Indented);
            File.WriteAllText(@"..\..\..\bounding_boxes.json", jsonOutput);
        }
    }
}