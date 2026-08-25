using System.Drawing;
using FaceID.App;

namespace FaceID.Tests;

/// <summary>
/// These run the real models over the sample images, so they are integration
/// tests rather than unit tests. Both models are committed to the repository,
/// which makes the numbers below stable and worth asserting on.
/// </summary>
[Collection("engine")]
public class FaceEngineTests(EngineFixture fixture)
{
    private FaceEngine Engine => fixture.Engine;

    [Fact]
    public void Analyze_FindsTheFaceInAPortrait()
    {
        FaceRecord? record = Engine.Analyze(TestData.Image("Aaron_Peirsol_0002.jpg"));

        Assert.NotNull(record);
        Assert.Equal("Aaron_Peirsol_0002.jpg", record.File);
        Assert.Equal(1, record.FaceCount);
        Assert.InRange(record.Subject.Score, 0.6f, 1.0f);
    }

    [Fact]
    public void Analyze_ProducesA128NumberEmbedding()
    {
        FaceRecord? record = Engine.Analyze(TestData.Image("Aaron_Peirsol_0002.jpg"));

        Assert.NotNull(record);
        Assert.Equal(128, record.Embedding.Length);
        Assert.Contains(record.Embedding, f => f != 0f);
    }

    /// <summary>
    /// Documents the trap that makes Cosine necessary: SFace does not return a
    /// unit vector, so a plain dot product would not be a cosine.
    /// </summary>
    [Fact]
    public void Analyze_EmbeddingIsNotUnitLength()
    {
        FaceRecord? record = Engine.Analyze(TestData.Image("Aaron_Peirsol_0002.jpg"));

        Assert.NotNull(record);
        Assert.InRange(Similarity.Length(record.Embedding), 5.0, 25.0);
    }

    [Fact]
    public void Analyze_IsDeterministic()
    {
        string path = TestData.Image("Adam_Sandler_0003.jpg");

        FaceRecord? first = Engine.Analyze(path);
        FaceRecord? second = Engine.Analyze(path);

        Assert.NotNull(first);
        Assert.NotNull(second);
        Assert.Equal(first.Embedding, second.Embedding);
    }

    [Fact]
    public void Analyze_OnAMissingFile_ReturnsNull()
    {
        Assert.Null(Engine.Analyze(TestData.Image("this_file_does_not_exist.jpg")));
    }

    [Fact]
    public void Analyze_OnAFileThatIsNotAnImage_ReturnsNull()
    {
        string path = Path.Combine(Path.GetTempPath(), $"faceid-not-an-image-{Guid.NewGuid():N}.jpg");
        File.WriteAllText(path, "this is text, not a JPEG");
        try
        {
            Assert.Null(Engine.Analyze(path));
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void Constructor_OnAMissingModel_Throws()
    {
        var error = Assert.Throws<FileNotFoundException>(() =>
            new FaceEngine("no-such-detector.onnx", TestData.RecognizerModel, 0.6f));

        Assert.Contains("no-such-detector.onnx", error.Message);
    }

    // ---------------------------------------------------------- recognition

    [Fact]
    public void TwoPhotosOfTheSamePerson_ScoreAboveTheDefaultThreshold()
    {
        double score = ScoreBetween("Abdullah_Gul_0003.jpg", "Abdullah_Gul_0004.jpg");

        Assert.True(score >= 0.371, $"same person scored only {score:F3}");
    }

    [Fact]
    public void TwoDifferentPeople_ScoreBelowTheDefaultThreshold()
    {
        double score = ScoreBetween("Abdullah_Gul_0003.jpg", "Adam_Sandler_0003.jpg");

        Assert.True(score < 0.371, $"different people scored {score:F3}");
    }

    [Theory]
    [InlineData("Adrien_Brody_0006.jpg", "Adrien_Brody_0007.jpg")]
    [InlineData("Adrien_Brody_0006.jpg", "Adrien_Brody_0010.jpg")]
    [InlineData("Al_Gore_0003.jpg", "Al_Gore_0006.jpg")]
    [InlineData("Alejandro_Toledo_0002.jpg", "Alejandro_Toledo_0005.jpg")]
    public void KnownSamePersonPairs_AreRecognised(string a, string b)
    {
        Assert.True(ScoreBetween(a, b) >= 0.371);
    }

    [Theory]
    [InlineData("Adam_Sandler_0003.jpg", "Al_Gore_0003.jpg")]
    [InlineData("Abdullah_Gul_0003.jpg", "Adrien_Brody_0006.jpg")]
    [InlineData("Ai_Sugiyama_0003.jpg", "Alan_Greenspan_0001.jpg")]
    public void KnownDifferentPeoplePairs_AreRejected(string a, string b)
    {
        Assert.True(ScoreBetween(a, b) < 0.371);
    }

    [Fact]
    public void APhotoComparedWithItself_ScoresOne()
    {
        FaceRecord? record = Engine.Analyze(TestData.Image("Al_Gore_0003.jpg"));

        Assert.NotNull(record);
        Assert.Equal(1.0, Similarity.Cosine(record.Embedding, record.Embedding), 5);
    }

    // ------------------------------------------------------ subject choice

    [Fact]
    public void ACrowdedImage_ReportsEveryFace()
    {
        FaceRecord? record = Engine.Analyze(TestData.Image("Abdel_Nasser_Assidi_0002.jpg"));

        Assert.NotNull(record);
        Assert.True(record.FaceCount > 1, "expected a bystander in this image");
    }

    /// <summary>
    /// The regression test for the subject rule. In this image the highest scoring
    /// face is a bystander at the edge, so picking by score enrols the wrong
    /// person. The subject has to be the face nearest the centre.
    /// </summary>
    [Fact]
    public void InACrowdedImage_TheSubjectIsTheCentralFace_NotTheHighestScoring()
    {
        string path = TestData.Image("Alejandro_Toledo_0028.jpg");
        FaceRecord? record = Engine.Analyze(path);

        Assert.NotNull(record);
        Assert.Equal(5, record.FaceCount);

        // 250x250 image; the subject's box centre must sit near the middle.
        PointF centre = record.Subject.Center;
        Assert.InRange(centre.X, 100f, 150f);
        Assert.InRange(centre.Y, 100f, 150f);

        // And it must still match other photos of the same man, which is the
        // behaviour that broke when selection went by score.
        FaceRecord? other = Engine.Analyze(TestData.Image("Alejandro_Toledo_0035.jpg"));
        Assert.NotNull(other);
        Assert.True(Similarity.Cosine(record.Embedding, other.Embedding) >= 0.371,
            "the crowded image enrolled the wrong face");
    }

    // ------------------------------------------------------------ geometry

    [Fact]
    public void FaceBox_ExposesCentreAndArea()
    {
        var box = new FaceBox(10, 20, 100, 200, 0.9f,
        [
            new PointF(30, 60), new PointF(90, 60), new PointF(60, 90),
            new PointF(40, 120), new PointF(80, 120),
        ]);

        Assert.Equal(60f, box.Center.X);
        Assert.Equal(120f, box.Center.Y);
        Assert.Equal(20000f, box.Area);
    }

    [Fact]
    public void FaceBox_EyeTiltIsZeroWhenTheEyesAreLevel()
    {
        var box = BoxWithEyes(new PointF(30, 60), new PointF(90, 60));
        Assert.Equal(0.0, box.EyeTilt, 6);
    }

    [Fact]
    public void FaceBox_EyeTiltIsFortyFiveDegreesOnADiagonal()
    {
        var box = BoxWithEyes(new PointF(30, 30), new PointF(90, 90));
        Assert.Equal(45.0, box.EyeTilt, 6);
    }

    [Fact]
    public void FaceBox_EyeTiltSignFollowsTheDirectionOfTilt()
    {
        Assert.True(BoxWithEyes(new PointF(30, 90), new PointF(90, 30)).EyeTilt < 0);
        Assert.True(BoxWithEyes(new PointF(30, 30), new PointF(90, 90)).EyeTilt > 0);
    }

    [Fact]
    public void RealPortraits_AreCloseToLevel()
    {
        FaceRecord? record = Engine.Analyze(TestData.Image("Aaron_Peirsol_0002.jpg"));

        Assert.NotNull(record);
        Assert.InRange(record.Subject.EyeTilt, -15.0, 15.0);
    }

    // ------------------------------------------------------------- helpers

    private double ScoreBetween(string fileA, string fileB)
    {
        FaceRecord? a = Engine.Analyze(TestData.Image(fileA));
        FaceRecord? b = Engine.Analyze(TestData.Image(fileB));

        Assert.NotNull(a);
        Assert.NotNull(b);
        return Similarity.Cosine(a.Embedding, b.Embedding);
    }

    private static FaceBox BoxWithEyes(PointF rightEye, PointF leftEye) =>
        new(0, 0, 120, 120, 0.9f,
            [rightEye, leftEye, new PointF(60, 90), new PointF(40, 110), new PointF(80, 110)]);
}
