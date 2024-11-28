public interface IOCRService
{
    string ExtractText(string imagePath);
    Task<string> ExtractTextAsync(string imagePath);
    void GenerateBoundingBoxesDataFile(string imagePath);
}
