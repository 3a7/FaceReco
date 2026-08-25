using FaceID.App;

namespace FaceID.Tests;

public class SimilarityTests
{
    [Fact]
    public void Cosine_OfAVectorWithItself_IsOne()
    {
        float[] v = [1f, -2f, 3f, 0.5f];
        Assert.Equal(1.0, Similarity.Cosine(v, v), 10);
    }

    [Fact]
    public void Cosine_OfPerpendicularVectors_IsZero()
    {
        Assert.Equal(0.0, Similarity.Cosine([1f, 0f], [0f, 1f]), 10);
    }

    [Fact]
    public void Cosine_OfOppositeVectors_IsMinusOne()
    {
        Assert.Equal(-1.0, Similarity.Cosine([1f, 2f], [-1f, -2f]), 10);
    }

    /// <summary>
    /// The one that matters. SFace embeddings are not unit length, so Cosine has
    /// to divide both norms out. If that division is ever dropped this test fails
    /// while the perpendicular and identical cases would still pass.
    /// </summary>
    [Fact]
    public void Cosine_IgnoresMagnitude()
    {
        float[] a = [1f, 2f, 3f];
        float[] b = [4f, 5f, 6f];
        float[] bScaledUp = [40f, 50f, 60f];

        Assert.Equal(Similarity.Cosine(a, b), Similarity.Cosine(a, bScaledUp), 10);
    }

    [Fact]
    public void Cosine_IsSymmetric()
    {
        float[] a = [0.3f, -1.4f, 2f];
        float[] b = [-0.7f, 0.2f, 5f];

        Assert.Equal(Similarity.Cosine(a, b), Similarity.Cosine(b, a), 10);
    }

    [Fact]
    public void Cosine_StaysWithinMinusOneAndOne()
    {
        var random = new Random(1234);
        for (int trial = 0; trial < 200; trial++)
        {
            float[] a = NextVector(random, 128);
            float[] b = NextVector(random, 128);
            double score = Similarity.Cosine(a, b);

            Assert.InRange(score, -1.0000001, 1.0000001);
        }
    }

    [Fact]
    public void Cosine_OnDifferentLengths_Throws()
    {
        var error = Assert.Throws<ArgumentException>(
            () => Similarity.Cosine([1f, 2f], [1f, 2f, 3f]));

        Assert.Contains("2", error.Message);
        Assert.Contains("3", error.Message);
    }

    [Theory]
    [InlineData(new[] { 3f, 4f }, 5.0)]
    [InlineData(new[] { 0f, 0f }, 0.0)]
    [InlineData(new[] { 1f, 1f, 1f, 1f }, 2.0)]
    public void Length_IsTheEuclideanNorm(float[] vector, double expected)
    {
        Assert.Equal(expected, Similarity.Length(vector), 10);
    }

    private static float[] NextVector(Random random, int size)
    {
        var v = new float[size];
        for (int i = 0; i < size; i++) v[i] = (float)(random.NextDouble() * 20 - 10);
        return v;
    }
}
