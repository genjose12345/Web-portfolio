import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

public class Assignment_4
{
    public static void main(String args[])
    {
        if(!(args.length == 1))
        {
            System.out.println("missing or too many command line arguments");
            System.out.println("Usage: assignment_4 <FileName>");
            return;
        }
        try(FileReader fileReader = new FileReader(args[0]); BufferedReader br = new BufferedReader(fileReader))
        {
            String line = br.readLine();
            Binary_Tree students = new Binary_Tree();
            while(line != null)
            {
                try
                {
                    String[] data = line.split(" ");
                    if(data.length < 5)
                    {
                        System.out.println("Improper data format: insufficient data elements");
                        line = br.readLine();
                    }

                    String name = data[0];
                    int id;
                    float gpa;

                    try 
                    {
                        id = Integer.parseInt(data[1]);
                        gpa = Float.parseFloat(data[2]);
                    } 
                    catch(NumberFormatException e)
                    {
                        System.out.println("Improper data format: invalid number format");
                        break;
                    }

                    if(data[3].equals("grad"))
                    {
                        graduate student = new graduate(name, id, gpa, data[4]);
                        students.insert(student);
                    }
                    else if(data[3].equals("undergrad"))
                    {
                        undergraduate student = new undergraduate(name, id, gpa, data[4]);
                        students.insert(student);
                    }
                    else
                    {
                        System.out.println("Improper data format: invalid student type");
                        line = br.readLine();
                    }
                }
                catch(IllegalArgumentException e)
                {
                    System.out.println("Improper data format: " + e.getMessage());
                }
                line = br.readLine();
            }
            students.inorderTransversal(students.root);
        }
        catch(IllegalArgumentException | IOException e)
        {
            System.out.println("Error reading file: " + e.getMessage());
        }
    }
}
