class Node <T extends Student>
{
    T data;
    Node<T> left;
    Node<T> right;

    Node(T data)
    {
        this.data = data;
        this.left = null;
        this.right = null;
    }
}

public class Binary_Tree<T extends Student>
{
    Node<T> root;

    public Node<T> insert(Node<T> node, T student)
    {
        if(student == null)
        {
            throw new IllegalArgumentException("Student cannot be null");
        }

        if(node == null)
        {

            if(root == null)
            {
                root = new Node<>(student);
                return root;
            }
            return new Node<>(student);
        }

        if(student.getId() < node.data.getId())
        {
            node.left = insert(node.left, student);
        }
        else if(student.getId() > node.data.getId())
        {
            node.right = insert(node.right, student);
        }
        else
        {
            throw new IllegalArgumentException("Student with ID " + student.getId() + " already exists");
        }
        return node;    
    }

    public Node<T> search(Node<T> node, int id)
    {
        if(node == null || node.data.getId() == id)
        {
            return node;
        }

        if(id < node.data.getId())
        {
            return search(node.left, id);
        }
        return search(node.right, id);
    }

    public void insert(T student)
    {
        root = insert(root, student);
    }

    public void inorderTransversal(Node node)
    {
        if(node == null)
        {
            return;
        }
        inorderTransversal(node.left);
        System.out.println(node.data + " ");
        inorderTransversal(node.right);
    }
}

