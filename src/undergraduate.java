public class undergraduate extends Student
{
    private Classification year;

    public undergraduate(String name, int id, float gpa, String year)
    {
        super(name, id, gpa);
        try 
        {
            this.year = Classification.valueOf(year.toLowerCase());
        } 
        catch (IllegalArgumentException e) 
        {
            throw new IllegalArgumentException("Invalid classification: " + year);
        }
    }

    @Override
    public boolean isOnProbation()
    {
        return (getGpa() < 2.0 );
    }

    public enum Classification
    {
        freshmen, sophomore, junior, senior;
    }

    @Override
    public String toString() 
    {
        return super.toString() + String.format(", Year: %s", year);
    }
}
