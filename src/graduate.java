public class graduate extends Student
{
    private String advisor;

    public String getAdvisor()
    {
        return advisor;
    }

    public void setAdvisor(String advisor)
    {
        if(advisor == null || advisor.trim().isEmpty())
        {
            throw new IllegalArgumentException("Advisor name cannot be empty");
        }
        this.advisor = advisor;
    }

    @Override
    public boolean isOnProbation()
    {
        return (getGpa() < 3.0 );
    }

    public graduate(String name, int id, float gpa, String advisor)
    {
        super(name, id, gpa);
        setAdvisor(advisor);
    }

    @Override
    public String toString() 
    {
        return super.toString() + String.format(", Advisor: %s", advisor);
    }
}
