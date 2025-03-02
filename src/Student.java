public abstract class Student
{
    private String name;
    private int id;
    private float gpa;

    public String getName()
    {
        return name;
    }

    public int getId()
    {
        return id;
    }

    public float getGpa()
    {
        return gpa;
    }

    public void setName(String name)
    {
        if(name == null || name.trim().isEmpty())
        {
            throw new IllegalArgumentException("Name cannot be empty");
        }
        this.name = name;
    }

    public void setId(int id)
    {
        if(Integer.toString(id).length() != 9)
        {
            throw new IllegalArgumentException("ID must be 9 digits");
        }
        this.id = id;
    }

    public void setGpa(float gpa)
    {
        if(gpa < 0.0 || gpa > 4.0)
        {
            throw new IllegalArgumentException("GPA must be between 0.0 and 4.0");
        }
        this.gpa = gpa;
    }

    public Student(String name, int id, float gpa)
    {
        setName(name);
        setId(id);
        setGpa(gpa);
    }

    public abstract boolean isOnProbation();

    @Override
    public String toString()
    {
        return String.format("Name: %s, ID: %d, GPA: %.2f, Probation: %s", 
            name, id, gpa, isOnProbation() ? "Yes" : "No");
    }
}
