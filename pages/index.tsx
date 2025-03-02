import React, { useState, useEffect, useRef } from 'react';
import type { NextPage } from 'next';
import { Github, Linkedin, Mail, ChevronDown, Award, Code, Database, Server, Cpu, ArrowRight, ExternalLink, X, MapPin, Calendar } from 'lucide-react';
import Layout from '../components/Layout';

// Types for our modals
type ProjectModalData = {
  title: string;
  description: string;
  technologies: string[];
  challenges: string;
  solution: string;
  results: string;
  imageGradient: string;
  year: string;
};

type BlogModalData = {
  title: string;
  date: string;
  category: string;
  content: string;
  imageGradient: string;
  description?: string; // Added optional description property
};

type TimelineItemData = {
  title: string;
  company: string;
  location: string;
  date: string;
  description: string[];
  details: string[];
  color: string;
  technologies: string[];
  skills: string[];
  impact?: string;
  teamSize?: string;
  tools?: string[];
};

const Portfolio: NextPage = () => {
  const [activeSection, setActiveSection] = useState<string>('home');
  const [isLoading, setIsLoading] = useState<boolean>(true);
  
  // Modal states
  const [isProjectModalOpen, setIsProjectModalOpen] = useState<boolean>(false);
  const [isBlogModalOpen, setIsBlogModalOpen] = useState<boolean>(false);
  const [isContactModalOpen, setIsContactModalOpen] = useState<boolean>(false);
  const [projectModalData, setProjectModalData] = useState<ProjectModalData | null>(null);
  const [blogModalData, setBlogModalData] = useState<BlogModalData | null>(null);
  const [isTimelineModalOpen, setIsTimelineModalOpen] = useState<boolean>(false);
  const [timelineModalData, setTimelineModalData] = useState<TimelineItemData | null>(null);
  
  // Sample project data
  const projectsData: ProjectModalData[] = [
    {
      title: "Network Traffic Predictor",
      description: "An advanced machine learning system designed to analyze network traffic patterns and predict future resource requirements.",
      technologies: ["AWS SageMaker", "Python", "ML Models", "AWS Lambda", "CloudWatch"],
      challenges: "Needed to develop efficient prediction models for network resource allocation while minimizing costs.",
      solution: "Developed ML models using AWS SageMaker to analyze network traffic patterns and predict future resource needs. Created Python scripts to automatically scale EC2 instances and allocate resources based on ML predictions.",
      results: "Implemented AWS CloudWatch monitoring and Lambda functions to optimize resource usage, reducing overall costs by 20%.",
      imageGradient: "from-blue-600 to-indigo-800",
      year: "2023"
    },
    {
      title: "Diabetic Retinopathy Detection",
      description: "A computer vision system that analyzes retinal images to detect early signs of diabetic retinopathy.",
      technologies: ["MATLAB", "AlexNet", "Transfer Learning", "Image Processing"],
      challenges: "Needed to develop a system for accurately detecting eye disease patterns in retinal images.",
      solution: "Utilized AlexNet model architecture to detect diabetic retinopathy through transfer learning techniques. Developed an automated system for analyzing retinal images to identify eye disease patterns.",
      results: "Achieved 98% accuracy in detecting diabetic retinopathy using transfer learning with AlexNet.",
      imageGradient: "from-purple-600 to-pink-500",
      year: "2023"
    },
    {
      title: "Cryptography Research Project",
      description: "A Java-based encryption system implementing novel cryptographic techniques for secure message transmission.",
      technologies: ["Java", "Cryptography", "Algorithm Design", "Security"],
      challenges: "Needed to develop a secure encryption program that could both encrypt and decrypt messages effectively.",
      solution: "Conducted undergraduate research in cryptography, developing a Java-based encryption program. Created a program that encrypts coded messages using a keyword, shifting letters based on the keyword's characters.",
      results: "Built a decryption module that analyzes encrypted messages, estimates the likely size of the keyword, and attempts to reconstruct the keyword by generating plausible sentences. Implemented a frequency analysis table to assist in identifying the keyword and decrypting the message.",
      imageGradient: "from-green-600 to-teal-500",
      year: "2022"
    },
    {
      title: "Carpet Pattern Classifier",
      description: "A computer vision system developed for Mohawk Industries to automatically classify carpet styles and patterns from images.",
      technologies: ["PyTorch", "Computer Vision", "Python", "OpenCV", "Data Pipeline"],
      challenges: "Carpet pattern classification involves subtle texture differences and complex pattern variations across thousands of product lines.",
      solution: "Developed deep learning computer vision model using PyTorch and Python to classify carpet styles and patterns. Processed and prepared large carpet image datasets for model training using Python and OpenCV.",
      results: "Built data pipeline and preprocessing workflow to transform raw images into ML-ready training data. Achieved 95% classification accuracy across multiple carpet styles and patterns.",
      imageGradient: "from-red-600 to-yellow-500",
      year: "2023"
    }
  ];
  
  // Sample blog data
  const blogsData: BlogModalData[] = [
    {
      title: "Transfer Learning Techniques for Computer Vision",
      date: "Jan 15, 2025",
      category: "Machine Learning",
      content: `
        <h2>Why Transfer Learning Matters in Computer Vision</h2>
        <p>Transfer learning has revolutionized how we approach computer vision problems, especially when working with limited datasets. Rather than training models from scratch, we can leverage pre-trained models that have already learned fundamental image features from massive datasets like ImageNet.</p>
        
        <h2>Key Benefits of Transfer Learning</h2>
        <p>The primary advantages of using transfer learning for computer vision tasks include:</p>
        <ul>
          <li><strong>Reduced Training Time:</strong> Pre-trained models already understand basic image features, so fine-tuning takes significantly less time than training from scratch.</li>
          <li><strong>Less Data Required:</strong> You can achieve excellent results with far fewer training examples—sometimes just hundreds instead of thousands or millions.</li>
          <li><strong>Better Performance:</strong> Starting with weights from a pre-trained model often leads to higher accuracy and faster convergence.</li>
        </ul>
        
        <h2>Popular Transfer Learning Techniques</h2>
        <p>When implementing transfer learning for computer vision, you typically have several approaches:</p>
        
        <h3>1. Feature Extraction</h3>
        <p>In this approach, you use the pre-trained model as a fixed feature extractor. You remove the final classification layer and extract the output from the penultimate layer as features for your new classifier.</p>
        <p>This works well when your new dataset is small or similar to the original dataset the model was trained on.</p>
        
        <h3>2. Fine-Tuning</h3>
        <p>Here, you not only replace the classifier but also fine-tune some of the weights of the pre-trained model by continuing backpropagation. Usually, earlier layers (which learn more generic features) are frozen, while later layers (which learn more specific features) are fine-tuned.</p>
        
        <h3>3. Progressive Fine-Tuning</h3>
        <p>This is a more nuanced approach where you gradually unfreeze layers from the top down as training progresses. This helps prevent catastrophic forgetting of the valuable features learned during pre-training.</p>
        
        <h2>Choosing the Right Pre-Trained Model</h2>
        <p>The choice of pre-trained model depends on your specific needs:</p>
        <ul>
          <li><strong>ResNet:</strong> Good general-purpose feature extractor with variants of different depths (ResNet-50, ResNet-101, etc.)</li>
          <li><strong>MobileNet:</strong> Optimized for mobile and edge devices</li>
          <li><strong>EfficientNet:</strong> State-of-the-art performance with efficient scaling</li>
          <li><strong>Vision Transformer (ViT):</strong> Transformer-based architecture showing promising results</li>
        </ul>
        
        <h2>Case Study: Diabetic Retinopathy Detection</h2>
        <p>In my work on diabetic retinopathy detection, I used transfer learning with AlexNet to classify retinal images. Despite having only 2,000 labeled images, we achieved 98% accuracy by:</p>
        <ol>
          <li>Using AlexNet pre-trained on ImageNet</li>
          <li>Replacing the final fully connected layers</li>
          <li>Fine-tuning the last few convolutional layers</li>
          <li>Implementing data augmentation to artificially expand our limited dataset</li>
        </ol>
        
        <h2>Best Practices for Transfer Learning in Computer Vision</h2>
        <p>To get the most out of transfer learning for your computer vision projects:</p>
        <ul>
          <li>Match preprocessing techniques to those used for the pre-trained model</li>
          <li>Start with a lower learning rate when fine-tuning</li>
          <li>Use appropriate data augmentation techniques</li>
          <li>Consider the similarity between your dataset and the pre-training dataset</li>
          <li>Experiment with freezing different numbers of layers</li>
        </ul>
        
        <h2>Conclusion</h2>
        <p>Transfer learning is not just a technique but a fundamental paradigm shift in how we approach computer vision problems. By leveraging the knowledge embedded in pre-trained models, we can build highly accurate systems with limited data and computational resources.</p>
        <p>As the field evolves, we're seeing more specialized pre-trained models for specific domains, making transfer learning even more powerful for specialized applications in healthcare, manufacturing, agriculture, and beyond.</p>
      `,
      imageGradient: "from-blue-600 to-purple-600"
    },
    {
      title: "Optimizing ML Deployments on AWS SageMaker",
      date: "Feb 2, 2025",
      category: "AWS",
      content: `
        <h2>The Challenge of ML Model Deployment</h2>
        <p>Deploying machine learning models to production environments presents unique challenges that go beyond the initial model development. From scaling issues to monitoring model drift, the operational aspects of ML can significantly impact both performance and cost.</p>
        
        <h2>Why AWS SageMaker for ML Deployments</h2>
        <p>Amazon SageMaker provides a comprehensive platform for building, training, and deploying machine learning models at scale. Key advantages include:</p>
        <ul>
          <li>End-to-end ML workflow support</li>
          <li>Built-in model optimization tools</li>
          <li>Flexible deployment options</li>
          <li>Automated scaling</li>
          <li>Integration with the broader AWS ecosystem</li>
        </ul>
        
        <h2>Optimizing Model Inference Performance</h2>
        
        <h3>Instance Selection Strategies</h3>
        <p>One of the most impactful decisions for both performance and cost is selecting the right instance type:</p>
        <ul>
          <li><strong>CPU Instances (C5, M5):</strong> Cost-effective for traditional ML algorithms and preprocessing</li>
          <li><strong>GPU Instances (G4, P3):</strong> Essential for deep learning model inference</li>
          <li><strong>Inf1 Instances:</strong> Optimized for cost-effective deep learning inference using AWS Inferentia chips</li>
        </ul>
        <p>For my network traffic prediction system, we initially deployed on g4dn.xlarge instances but later switched to inf1.xlarge, reducing inference costs by 45% while maintaining performance.</p>
        
        <h3>Multi-Model Endpoints</h3>
        <p>When deploying multiple related models, SageMaker's multi-model endpoints allow you to host several models on a single endpoint. This significantly reduces costs for applications that don't need all models to be constantly available at high capacity.</p>
        
        <h2>Optimizing for Cost Efficiency</h2>
        
        <h3>Autoscaling Configuration</h3>
        <p>Properly configured autoscaling is crucial for balancing performance and cost:</p>
        <ol>
          <li>Set appropriate minimum and maximum instance counts</li>
          <li>Choose the right scaling metric (typically InvocationsPerInstance)</li>
          <li>Configure scale-in and scale-out cooldown periods to prevent thrashing</li>
          <li>Use target tracking scaling based on expected traffic patterns</li>
        </ol>
        
        <h3>Serverless Inference</h3>
        <p>For workloads with unpredictable or intermittent traffic, SageMaker Serverless Inference automatically scales capacity based on traffic and eliminates the need to select instance types or manage scaling policies.</p>
        
        <h3>Batch Transform for Bulk Predictions</h3>
        <p>For non-real-time predictions, Batch Transform jobs are significantly more cost-effective than maintaining always-on endpoints. We reduced costs by 75% by shifting our daily prediction workloads to scheduled batch jobs.</p>
        
        <h2>Deployment Architecture Best Practices</h2>
        
        <h3>Model Monitoring and Drift Detection</h3>
        <p>Set up SageMaker Model Monitor to automatically detect:
        <ul>
          <li>Data quality issues in production inputs</li>
          <li>Model drift as real-world data evolves</li>
          <li>Bias introduction in production data</li>
        </ul>
        </p>
        
        <h3>Feature Store for Consistent Features</h3>
        <p>Use SageMaker Feature Store to ensure consistency between training and inference feature engineering. This eliminates one of the most common sources of production ML issues.</p>
        
        <h3>Model A/B Testing with Production Variants</h3>
        <p>When rolling out model updates, use production variants to gradually shift traffic from the existing model to the new version, monitoring performance metrics to ensure the new model performs as expected in production.</p>
        
        <h2>Containerization and CI/CD for ML</h2>
        <p>Implement a robust CI/CD pipeline that automates:
        <ul>
          <li>Model retraining with new data</li>
          <li>Performance validation against benchmarks</li>
          <li>Deployment with automated rollbacks if performance degrades</li>
        </ul>
        </p>
        
        <h2>Conclusion: The Evolving ML Deployment Landscape</h2>
        <p>AWS SageMaker continues to evolve with new capabilities like SageMaker Clarify for explainability, SageMaker Edge Manager for edge deployments, and SageMaker Neo for model optimization.</p>
        <p>By following these best practices and leveraging SageMaker's managed infrastructure, you can focus on model quality and business outcomes rather than operational complexities.</p>
      `,
      imageGradient: "from-green-600 to-blue-600"
    },
    {
      title: "Building Custom Loss Functions in PyTorch",
      date: "Feb 20, 2025",
      category: "PyTorch",
      content: `
        <h2>Beyond Standard Loss Functions</h2>
        <p>While PyTorch provides many common loss functions like Cross Entropy Loss and Mean Squared Error, custom loss functions can be crucial for specialized tasks. They allow you to encode domain-specific knowledge and optimization objectives directly into your training process.</p>
        
        <h2>Why Create Custom Loss Functions?</h2>
        <p>Standard loss functions may not capture the nuances of your specific problem:</p>
        <ul>
          <li>Class imbalance issues not addressed by standard losses</li>
          <li>Domain-specific performance metrics that should guide training</li>
          <li>Multi-objective optimization requirements</li>
          <li>Special regularization needs</li>
        </ul>
        
        <h2>Implementing Custom Losses in PyTorch</h2>
        <p>Creating custom loss functions in PyTorch is straightforward thanks to its dynamic computational graph architecture.</p>
        
        <h3>Method 1: Function-Based Approach</h3>
        <pre>
def focal_loss(predictions, targets, alpha=0.25, gamma=2.0):
    """
    Focal Loss implementation for addressing class imbalance
    """
    BCE_loss = F.binary_cross_entropy_with_logits(
        predictions, targets, reduction='none'
    )
    pt = torch.exp(-BCE_loss)  # prevents nans when probability 0
    focal_loss = alpha * (1-pt)**gamma * BCE_loss
    return focal_loss.mean()
        </pre>
        
        <h3>Method 2: Class-Based Approach</h3>
        <pre>
class DiceLoss(nn.Module):
    """
    Dice Loss for image segmentation tasks
    """
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        
    def forward(self, predictions, targets):
        # Flatten predictions and targets
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        
        intersection = (predictions * targets).sum()
        dice = (2. * intersection + self.smooth) / (
            predictions.sum() + targets.sum() + self.smooth
        )
        return 1 - dice
        </pre>
        
        <h2>Case Study: Custom Loss for Carpet Pattern Detection</h2>
        <p>For my carpet pattern classifier at Mohawk Industries, I developed a custom loss function that combined:</p>
        <ol>
          <li>Cross entropy for basic classification</li>
          <li>Structure-based regularization to account for pattern symmetry</li>
          <li>Texture consistency penalties</li>
        </ol>
        
        <pre>
class CarpetPatternLoss(nn.Module):
    def __init__(self, texture_weight=0.3, symmetry_weight=0.2):
        super(CarpetPatternLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.texture_weight = texture_weight
        self.symmetry_weight = symmetry_weight
        
    def forward(self, predictions, targets, feature_maps):
        # Base classification loss
        ce = self.ce_loss(predictions, targets)
        
        # Texture consistency term
        texture_loss = self.calculate_texture_consistency(feature_maps)
        
        # Symmetry regularization
        symmetry_loss = self.calculate_symmetry_deviation(feature_maps)
        
        # Combined loss
        total_loss = ce + (self.texture_weight * texture_loss) + \
                     (self.symmetry_weight * symmetry_loss)
        return total_loss
        </pre>
        
        <p>This custom loss improved classification accuracy by 8% compared to standard cross entropy, particularly for patterns with subtle variations.</p>
        
        <h2>Advanced Techniques for Custom Losses</h2>
        
        <h3>Curriculum Learning with Dynamic Losses</h3>
        <p>Implement losses that change during training to gradually increase difficulty:</p>
        <pre>
class CurriculumLoss(nn.Module):
    def __init__(self, max_epochs):
        super(CurriculumLoss, self).__init__()
        self.max_epochs = max_epochs
        
    def forward(self, predictions, targets, epoch):
        # Calculate difficulty factor based on current epoch
        difficulty = min(1.0, epoch / (0.7 * self.max_epochs))
        
        # Apply weighted loss components based on difficulty
        # ...
        </pre>
        
        <h3>Adversarial Losses</h3>
        <p>For generative models, adversarial losses pit a generator against a discriminator, pushing the generator to produce increasingly realistic outputs.</p>
        
        <h3>Perceptual Losses</h3>
        <p>Using feature activations from pretrained networks (like VGG) to calculate losses that better align with human visual perception.</p>
        
        <h2>Debugging Custom Loss Functions</h2>
        <p>Custom losses can introduce subtle bugs. Key debugging strategies include:</p>
        <ul>
          <li>Gradual implementation: Start with standard loss and add custom components incrementally</li>
          <li>Gradient checking to verify backpropagation</li>
          <li>Tracking loss components separately during training</li>
          <li>Testing with simplified edge cases</li>
        </ul>
        
        <h2>Conclusion</h2>
        <p>Custom loss functions are powerful tools for incorporating domain knowledge into your neural networks. By moving beyond standard losses, you can address specific challenges in your datasets and significantly improve model performance.</p>
        <p>The key is to ensure your custom loss function is differentiable, numerically stable, and correctly aligns with your ultimate performance metrics.</p>
      `,
      imageGradient: "from-yellow-500 to-red-600"
    }
  ];
  
  // Loading effect
  useEffect(() => {
    const timer = setTimeout(() => {
      setIsLoading(false);
    }, 1500);
    
    return () => clearTimeout(timer);
  }, []);
  
  // Update the animation code to be simpler and more reliable
  useEffect(() => {
    // Force all sections to be visible immediately
    document.querySelectorAll('.animate-section').forEach(section => {
      section.classList.add('force-visible');
    });

    // Setup animation triggering
    const handleScroll = () => {
      document.querySelectorAll('.animate-section').forEach(section => {
        const sectionTop = section.getBoundingClientRect().top;
        const windowHeight = window.innerHeight;
        
        // If section is in viewport
        if (sectionTop < windowHeight * 0.75) {
          section.classList.add('animate-visible');
          
          // Animate children with delay
          section.querySelectorAll('.animate-item').forEach((item, index) => {
            setTimeout(() => {
              item.classList.add('item-visible');
            }, 300 + (index * 150));
          });
        }
      });
    };
    
    // Trigger once on load to animate initial content
    handleScroll();
    
    // Add scroll listener
    window.addEventListener('scroll', handleScroll);
    
    // Cleanup
    return () => {
      window.removeEventListener('scroll', handleScroll);
    };
  }, []);
  
  // Smooth scrolling function
  const scrollToSection = (sectionId: string) => {
    setActiveSection(sectionId);
    
    const element = document.getElementById(sectionId);
    if (element) {
      // Calculate the navbar height to offset the scroll position
      const navbarHeight = 80; // Approximate height of navbar
      
      const elementPosition = element.getBoundingClientRect().top;
      const offsetPosition = elementPosition + window.pageYOffset - navbarHeight;
      
      window.scrollTo({
        top: offsetPosition,
        behavior: "smooth"
      });
    }
  };
  
  // Open project modal
  const openProjectModal = (project: ProjectModalData) => {
    setProjectModalData(project);
    setIsProjectModalOpen(true);
    document.body.style.overflow = 'hidden'; // Prevent scrolling when modal is open
  };
  
  // Open blog modal
  const openBlogModal = (blog: BlogModalData) => {
    setBlogModalData(blog);
    setIsBlogModalOpen(true);
    document.body.style.overflow = 'hidden';
  };
  
  // Close modals
  const closeModals = () => {
    setIsProjectModalOpen(false);
    setIsBlogModalOpen(false);
    setIsContactModalOpen(false);
    setIsTimelineModalOpen(false);
    document.body.style.overflow = 'auto'; // Re-enable scrolling
  };
  
  // Work experience data
  const timelineData: TimelineItemData[] = [
    {
      title: "Machine Learning Intern",
      company: "Mohawk Industries",
      location: "Dalton, GA",
      date: "Summer 2023",
      description: [
        "Developed deep learning computer vision model using PyTorch and Python to classify carpet styles and patterns with 95% accuracy",
        "Processed and prepared large carpet image datasets for model training using Python and OpenCV",
        "Built data pipeline and preprocessing workflow to transform raw images into ML-ready training data"
      ],
      color: "blue",
      details: [
        "Developed deep learning computer vision model using PyTorch and Python to classify carpet styles and patterns with 95% accuracy",
        "Processed and prepared large carpet image datasets for model training using Python and OpenCV",
        "Built data pipeline and preprocessing workflow to transform raw images into ML-ready training data"
      ],
      technologies: ["Python", "PyTorch", "OpenCV", "Deep Learning", "Computer Vision"],
      skills: ["Machine Learning", "Data Preprocessing", "Model Training", "Computer Vision"],
      impact: "Created a computer vision system that significantly improved carpet pattern classification accuracy",
      tools: ["PyTorch", "Python", "Jupyter Notebooks", "Git"]
    },
    {
      title: "UGA Hackathon 2025",
      company: "University of Georgia",
      location: "Athens, GA",
      date: "Fall 2022",
      description: [
        "Engineered data analytics solution using ECL (Enterprise Control Language) to efficiently query and process large datasets on a distributed computing cluster",
        "Built responsive React/TypeScript web application to visualize cluster data through interactive charts and graphs, styled with Tailwind CSS",
        "Implemented real-time data visualization components including line charts, heatmaps, and statistical dashboards to present cluster analytics"
      ],
      color: "purple",
      details: [
        "Engineered data analytics solution using ECL (Enterprise Control Language) to efficiently query and process large datasets on a distributed computing cluster",
        "Built responsive React/TypeScript web application to visualize cluster data through interactive charts and graphs, styled with Tailwind CSS",
        "Implemented real-time data visualization components including line charts, heatmaps, and statistical dashboards to present cluster analytics"
      ],
      technologies: ["React", "TypeScript", "ECL", "Tailwind CSS", "Data Visualization"],
      skills: ["Frontend Development", "Data Analytics", "UI/UX Design", "Visualization"],
      teamSize: "3 members"
    },
    {
      title: "IS/Developer Intern",
      company: "Mohawk Industries",
      location: "Dalton, GA",
      date: "Summer 2022",
      description: [
        "Developed and maintained a critical software tool for international deployment using Java, JavaFX and SQL databases",
        "Created and optimized SQL queries for efficient data management across large, international datasets",
        "Developed API endpoint integrating HTTP requests and SQL queries for database interactions"
      ],
      color: "green",
      details: [
        "Developed and maintained a critical software tool for international deployment using Java, JavaFX and SQL databases",
        "Created and optimized SQL queries for efficient data management across large, international datasets",
        "Developed API endpoint integrating HTTP requests and SQL queries for database interactions"
      ],
      technologies: ["Java", "JavaFX", "SQL", "API Development", "Database Management"],
      skills: ["Backend Development", "Database Design", "API Integration", "SQL Query Optimization"],
      tools: ["Java", "JavaFX", "SQL", "Git", "REST APIs"]
    },
    {
      title: "Job Shadow - Senior Java Systems Developer",
      company: "Shaw Industries",
      location: "Dalton, GA",
      date: "Fall 2021",
      description: [
        "Observed and learned industry-standard Java development practices and system architecture design",
        "Gained insights into Agile methodologies and collaborative software development processes",
        "Participated in code reviews and debugging sessions, enhancing problem-solving skills"
      ],
      color: "red",
      details: [
        "Observed and learned industry-standard Java development practices and system architecture design",
        "Gained insights into Agile methodologies and collaborative software development processes",
        "Participated in code reviews and debugging sessions, enhancing problem-solving skills"
      ],
      technologies: ["Java", "Agile", "System Architecture", "Software Development"],
      skills: ["Code Review", "Problem Solving", "Software Development Lifecycle", "Team Collaboration"]
    }
  ];
  
  // Loading screen
  if (isLoading) {
    return (
      <div className="fixed inset-0 flex items-center justify-center bg-gray-900">
        <div className="text-center">
          <div className="text-5xl font-bold text-blue-500 mb-4">JR</div>
          <div className="relative w-64 h-2 bg-gray-700 rounded-full overflow-hidden">
            <div className="absolute top-0 left-0 h-full bg-blue-500 animate-loading-bar" style={{width: '100%'}}></div>
          </div>
        </div>
      </div>
    );
  }
  
  return (
    <Layout>
      {/* Contact Button */}
      <div className="fixed bottom-6 right-6 z-[100]">
        <button 
          onClick={() => setIsContactModalOpen(true)}
          className="contact-button bg-blue-600 hover:bg-blue-700 text-white p-4 rounded-full shadow-lg transition-all duration-300 flex items-center justify-center transform hover:scale-110"
        >
          <Mail size={20} className="animate-bounce" />
          <span className="absolute w-full h-full rounded-full bg-blue-500 animate-ping opacity-20"></span>
        </button>
      </div>
      
      {/* Navigation */}
      <nav className="fixed top-0 left-0 right-0 bg-gray-900 bg-opacity-90 z-50 border-b border-gray-800">
        <div className="container mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="text-2xl font-bold text-blue-500">Jose Rodriguez</div>
            <div className="hidden md:flex space-x-8">
              {['home', 'about', 'skills', 'projects', 'blog'].map((item) => (
                <button 
                  key={item}
                  className={`relative px-1 py-2 transition-colors ${activeSection === item ? 'text-blue-400' : 'text-gray-300 hover:text-white'}`}
                  onClick={() => scrollToSection(item)}
                >
                  {item.charAt(0).toUpperCase() + item.slice(1)}
                  {activeSection === item && (
                    <span className="absolute bottom-0 left-0 w-full h-0.5 bg-blue-500"></span>
                  )}
                </button>
              ))}
            </div>
            <div className="flex space-x-4">
              <a href="https://github.com/genjose12345" target="_blank" rel="noopener noreferrer" className="text-gray-400 hover:text-white transition-colors">
                <Github size={20} />
              </a>
              <a href="https://www.linkedin.com/in/jose-rodriguez-9a982b224/" target="_blank" rel="noopener noreferrer" className="text-gray-400 hover:text-white transition-colors">
                <Linkedin size={20} />
              </a>
              <a href="mailto:genjose1231@gmail.com" className="text-gray-400 hover:text-white transition-colors">
                <Mail size={20} />
              </a>
            </div>
          </div>
        </div>
      </nav>
      
      {/* Hero Section with enhanced animations */}
      <section id="home" className="min-h-screen flex items-center relative overflow-hidden animated-hero-bg">
        {/* Floating decorative elements */}
        <div className="floating-shape shape1"></div>
        <div className="floating-shape shape2"></div>
        <div className="floating-shape shape3"></div>
        <div className="floating-shape shape4"></div>
        
        <div className="container mx-auto px-6 relative z-10">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-center">
            <div className="text-center lg:text-left">
              <h1 className="text-5xl font-bold mb-4 hero-title">
                <span className="inline-block glitch-text">Hi, I'm </span>
                <span className="inline-block gradient-animated-text ml-2">Jose Rodriguez</span>
              </h1>
              <h2 className="text-2xl font-semibold mb-6 typewriter-text">Machine Learning Engineer</h2>
              <p className="text-xl text-gray-300 mb-10 fade-in-up">
                Specializing in AI/ML systems with extensive experience in Python, TensorFlow, PyTorch, and AWS Cloud infrastructure.
              </p>
              
              <div className="flex flex-wrap gap-4 justify-center lg:justify-start">
                <button 
                  onClick={() => scrollToSection('projects')}
                  className="magic-btn px-6 py-3 rounded-lg font-bold relative overflow-hidden"
                >
                  <span className="relative z-10">View Projects</span>
                </button>
                <button 
                  onClick={() => scrollToSection('about')}
                  className="outline-btn px-6 py-3 rounded-lg font-bold border border-blue-400 text-blue-400 hover:bg-blue-400/10 transition-colors"
                >
                  About Me
                </button>
              </div>
            </div>
            
            <div className="code-container perspective-container">
              <pre className="language-javascript code-block-animated rounded-xl p-6 relative overflow-hidden">
                <div className="code-typing-cursor"></div>
                <code className="text-gray-300">
                  <span className="text-pink-400">class</span> <span className="text-yellow-400">JoseRodriguez</span> <span className="text-pink-400">extends</span> <span className="text-yellow-400">Engineer</span> {'{'}
                  <br />
                  <br />
                  {'  '}<span className="text-blue-400">skills</span> = [
                  <span className="text-green-400 skill-tag">'Machine Learning'</span>, 
                  <span className="text-green-400 skill-tag">'Python'</span>, 
                  <span className="text-green-400 skill-tag">'AWS'</span>,
                  <br />
                  {'           '}<span className="text-green-400 skill-tag">'TensorFlow'</span>, 
                  <span className="text-green-400 skill-tag">'PyTorch'</span>, 
                  <span className="text-green-400 skill-tag">'React'</span>];
                  <br />
                  <br />
                  {'  '}<span className="text-purple-400">constructor</span>() {'{'}
                  <br />
                  {'    '}<span className="text-pink-400">super</span>();
                  <br />
                  {'    '}<span className="text-blue-400">this</span>.education = <span className="text-green-400">'B.S. Computer Science'</span>;
                  <br />
                  {'    '}<span className="text-blue-400">this</span>.location = <span className="text-green-400">'Dalton, GA'</span>;
                  <br />
                  {'  '}{'}'};
                  <br />
                  <br />
                  {'  '}<span className="text-purple-400">getCurrentFocus</span>() {'{'}
                  <br />
                  {'    '}<span className="text-pink-400">return</span> <span className="text-green-400 focus-text">'Advanced ML models & Cloud Architecture'</span>;
                  <br />
                  {'  '}{'}'};
                  <br />
                  {'}'}
                </code>
              </pre>
              <div className="code-reflection"></div>
            </div>
          </div>
        </div>
        
        <div className="scroll-indicator">
          <span>Scroll Down</span>
          <ChevronDown size={20} className="animate-bounce mt-2" />
        </div>
      </section>
      
      {/* About Section - Enhanced with animations */}
      <section id="about" className="py-20 animate-section bg-gray-800 relative">
        <div className="floating-element section-element-1"></div>
        <div className="floating-element section-element-2"></div>
        <div className="accent-pulse w-64 h-64 top-20 right-20"></div>
        <div className="accent-pulse w-96 h-96 bottom-20 left-10" style={{animationDelay: '3s'}}></div>

        <div className="container mx-auto px-6 relative z-10">
          <h2 className="text-3xl font-bold mb-12 text-center animate-item gradient-text">About Me</h2>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-12">
            <div className="animate-item">
              <p className="text-lg text-gray-300 mb-6 hover:text-white transition-colors duration-300">
                Specializing in AI/ML systems with extensive experience in Python, TensorFlow, PyTorch, and AWS Cloud infrastructure.
              </p>
              <p className="text-lg text-gray-300 mb-6 hover:text-white transition-colors duration-300">
                My journey in tech began with a fascination for software development and has evolved into a career focused on applying AI. I'm currently exploring opportunities where I can leverage my technical skills.
              </p>
              
              {/* Education - Enhanced styling */}
              <div className="about-card bg-gray-700/30 rounded-xl p-5 backdrop-blur-sm mb-6 border border-transparent hover:border-blue-500/30">
                <h3 className="text-xl font-bold mb-2 flex items-center">
                  <Award className="mr-2 text-blue-400" size={24} /> 
                  <span className="gradient-text">Education</span>
                </h3>
                <div className="flex items-start">
                  <div className="mt-1">
                    <div className="w-3 h-3 rounded-full bg-blue-500 mr-3 animate-pulse"></div>
                  </div>
                  <div>
                    <h4 className="font-bold">BS in Computer Science</h4>
                    <p className="text-gray-400">University of Kennesaw State with a concentration in Artificial Intelligence</p>
                  </div>
                </div>
              </div>
              
              {/* Stats - Enhanced with animations */}
              <div className="grid grid-cols-3 gap-4 mb-6">
                <div className="stats-card bg-gray-700/30 rounded-xl p-4 backdrop-blur-sm text-center border border-transparent hover:border-blue-500/30">
                  <div className="stats-number text-3xl font-bold text-blue-400">1+ Years</div>
                  <div className="text-gray-400 text-sm">Engineering Experience</div>
                </div>
                <div className="stats-card bg-gray-700/30 rounded-xl p-4 backdrop-blur-sm text-center border border-transparent hover:border-blue-500/30">
                  <div className="stats-number text-3xl font-bold text-blue-400">5+</div>
                  <div className="text-gray-400 text-sm">Projects Completed</div>
                </div>
                <div className="stats-card bg-gray-700/30 rounded-xl p-4 backdrop-blur-sm text-center border border-transparent hover:border-blue-500/30">
                  <div className="stats-number text-3xl font-bold text-blue-400">16+</div>
                  <div className="text-gray-400 text-sm">Technologies</div>
                </div>
              </div>
            </div>
            
            {/* About image with enhanced animations */}
            <div className="animate-item">
              <div className="bg-gradient-to-br from-blue-600/20 to-purple-600/20 rounded-xl p-6 h-full flex items-center justify-center transform hover:scale-[1.02] transition-transform about-card">
                <div className="text-center">
                  <div className="mb-8 w-32 h-32 mx-auto rounded-full bg-blue-500/20 flex items-center justify-center transform hover:rotate-12 transition-transform">
                    <Code size={64} className="text-blue-400 animate-pulse" />
                  </div>
                  <p className="text-xl text-gray-300 mb-3 gradient-text">Machine Learning Engineer</p>
                  <p className="text-gray-400">Specialized in AI/ML Solutions</p>
                  <div className="mt-6 flex justify-center space-x-6">
                    <div className="text-blue-400 floating-icon hover:scale-110 transition-transform cursor-pointer">
                      <Database className="w-8 h-8 mx-auto mb-2" />
                      <p className="text-xs">Data</p>
                    </div>
                    <div className="text-purple-400 floating-icon hover:scale-110 transition-transform cursor-pointer">
                      <Cpu className="w-8 h-8 mx-auto mb-2" />
                      <p className="text-xs">AI/ML</p>
                    </div>
                    <div className="text-cyan-400 floating-icon hover:scale-110 transition-transform cursor-pointer">
                      <Server className="w-8 h-8 mx-auto mb-2" />
                      <p className="text-xs">Cloud</p>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>
      
      {/* Work Experience Timeline */}
      <section id="experience" className="py-20 animate-section bg-gray-800" style={{opacity: 1, visibility: 'visible'}}>
        <div className="floating-element section-element-1" style={{background: 'rgba(16, 185, 129, 0.3)'}}></div>
        <div className="floating-element section-element-2" style={{background: 'rgba(59, 130, 246, 0.3)'}}></div>
        <div className="container mx-auto px-6">
          <h2 className="text-3xl font-bold mb-12 text-center animate-item">Experience Timeline</h2>
          {/* Timeline Component */}
          <div className="relative animate-item">
            {/* Timeline line */}
            <div className="absolute left-1/2 transform -translate-x-1/2 w-1 h-full bg-gray-700 timeline-line"></div>
            
            {/* Timeline items */}
            <div className="space-y-16">
              {timelineData.map((item, index) => (
                <div key={index} className={`relative flex items-center ${index % 2 === 0 ? 'justify-start' : 'justify-end'} animate-item`}>
                  {/* Timeline dot */}
                  <div className={`absolute left-1/2 transform -translate-x-1/2 w-4 h-4 rounded-full bg-${item.color}-500 timeline-dot z-10`}></div>
                  
                  {/* Content card */}
                  <div 
                    className={`w-5/12 work-card card-hover-effect bg-gray-800/40 rounded-xl p-6 relative group cursor-pointer`}
                    onClick={() => {
                      setTimelineModalData(item);
                      setIsTimelineModalOpen(true);
                    }}
                  >
                    <div className="space-y-4">
                      <div className={`bg-${item.color}-500/20 p-3 rounded-xl inline-block work-icon-container`}>
                        <Award className={`text-${item.color}-400 w-6 h-6 work-icon`} />
                      </div>
                      <div>
                        <h3 className={`text-xl font-bold text-${item.color}-400`}>{item.title}</h3>
                        <p className="text-lg text-gray-300">{item.company}</p>
                        <div className="flex items-center gap-4 text-sm text-gray-400 mt-2">
                          <span className="flex items-center gap-1">
                            <Calendar size={14} />
                            {item.date}
                          </span>
                          <span className="flex items-center gap-1">
                            <MapPin size={14} />
                            {item.location}
                          </span>
                        </div>
                      </div>
                      <div className="flex flex-wrap gap-2">
                        {item.technologies.slice(0, 3).map((tech, i) => (
                          <span 
                            key={i} 
                            className={`px-2 py-1 bg-${item.color}-500/10 text-${item.color}-400 rounded-lg text-xs work-tag`}>
                            {tech}
                          </span>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>
      
      {/* Skills Section */}
      <section id="skills" className="py-20 animate-section bg-gray-800" style={{opacity: 1, visibility: 'visible'}}>
        <div className="floating-element section-element-1" style={{background: 'rgba(236, 72, 153, 0.3)'}}></div>
        <div className="floating-element section-element-2" style={{background: 'rgba(251, 191, 36, 0.3)'}}></div>
        <div className="container mx-auto px-6">
          <h2 className="text-3xl font-bold mb-12 text-center animate-item">Skills & Expertise</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-12">
            {/* Machine Learning Skill Card */}
            <div className="skill-card card-hover-effect bg-gray-800/40 rounded-xl overflow-hidden p-6 relative group animate-item">
              <div className="relative z-10">
                <div className="bg-blue-500/20 p-4 rounded-xl inline-block mb-4 skill-icon-container">
                  <Cpu className="text-blue-400 w-8 h-8 skill-icon" />
                </div>
                <h3 className="text-xl font-bold mb-3 text-blue-400/90 group-hover:text-blue-400">Machine Learning</h3>
                <p className="text-gray-400 text-sm mb-4">
                  Deep learning, neural networks, and predictive modeling experience with various frameworks and libraries.
                </p>
                <div className="flex flex-wrap gap-2">
                  <span className="px-3 py-1 bg-blue-500/10 text-blue-400/90 rounded-lg text-xs">PyTorch</span>
                  <span className="px-3 py-1 bg-blue-500/10 text-blue-400/90 rounded-lg text-xs">TensorFlow</span>
                  <span className="px-3 py-1 bg-blue-500/10 text-blue-400/90 rounded-lg text-xs">scikit-learn</span>
                </div>
              </div>
            </div>
            
            {/* Programming Skill Card */}
            <div className="skill-card card-hover-effect bg-gray-800/40 rounded-xl overflow-hidden p-6 relative group animate-item">
              <div className="relative z-10">
                <div className="bg-green-500/20 p-4 rounded-xl inline-block mb-4 skill-icon-container">
                  <Code className="text-green-400 w-8 h-8 skill-icon" />
                </div>
                <h3 className="text-xl font-bold mb-3 text-green-400/90 group-hover:text-green-400">Programming</h3>
                <p className="text-gray-400 text-sm mb-4">
                  Full-stack development with modern frameworks and languages. Experienced in building scalable web applications.
                </p>
                <div className="flex flex-wrap gap-2">
                  <span className="px-3 py-1 bg-green-500/10 text-green-400/90 rounded-lg text-xs">React</span>
                  <span className="px-3 py-1 bg-green-500/10 text-green-400/90 rounded-lg text-xs">Python</span>
                  <span className="px-3 py-1 bg-green-500/10 text-green-400/90 rounded-lg text-xs">Java</span>
                </div>
              </div>
            </div>
            
            {/* Data Science Skill Card */}
            <div className="skill-card card-hover-effect bg-gray-800/40 rounded-xl overflow-hidden p-6 relative group animate-item">
              <div className="relative z-10">
                <div className="bg-purple-500/20 p-4 rounded-xl inline-block mb-4 skill-icon-container">
                  <Database className="text-purple-400 w-8 h-8 skill-icon" />
                </div>
                <h3 className="text-xl font-bold mb-3 text-purple-400/90 group-hover:text-purple-400">Data Science</h3>
                <p className="text-gray-400 text-sm mb-4">
                  Data processing and visualization expertise. Experienced in extracting insights from complex datasets.
                </p>
                <div className="flex flex-wrap gap-2">
                  <span className="px-3 py-1 bg-purple-500/10 text-purple-400/90 rounded-lg text-xs">Pandas & NumPy</span>
                  <span className="px-3 py-1 bg-purple-500/10 text-purple-400/90 rounded-lg text-xs">Matplotlib</span>
                  <span className="px-3 py-1 bg-purple-500/10 text-purple-400/90 rounded-lg text-xs">SQL</span>
                </div>
              </div>
            </div>
            
            {/* Cloud Skills Card */}
            <div className="skill-card card-hover-effect bg-gray-800/40 rounded-xl overflow-hidden p-6 relative group animate-item">
              <div className="relative z-10">
                <div className="bg-indigo-500/20 p-4 rounded-xl inline-block mb-4 skill-icon-container">
                  <Server className="text-indigo-400 w-8 h-8 skill-icon" />
                </div>
                <h3 className="text-xl font-bold mb-3 text-indigo-400/90 group-hover:text-indigo-400">AWS Cloud</h3>
                <p className="text-gray-400 text-sm mb-4">
                  Experience with AWS services for ML deployments. Skilled in cloud infrastructure management.
                </p>
                <div className="flex flex-wrap gap-2">
                  <span className="px-3 py-1 bg-indigo-500/10 text-indigo-400/90 rounded-lg text-xs">SageMaker</span>
                  <span className="px-3 py-1 bg-indigo-500/10 text-indigo-400/90 rounded-lg text-xs">Lambda & EC2</span>
                  <span className="px-3 py-1 bg-indigo-500/10 text-indigo-400/90 rounded-lg text-xs">S3 & DynamoDB</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>
      
      {/* Projects Section */}
      <section id="projects" className="py-20 animate-section bg-gray-900">
        <div className="floating-element section-element-1" style={{background: 'rgba(56, 189, 248, 0.3)'}}></div>
        <div className="floating-element section-element-2" style={{background: 'rgba(167, 139, 250, 0.3)'}}></div>
        <div className="container mx-auto px-6">
          <h2 className="text-3xl font-bold mb-12 text-center animate-item">Featured Projects</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 animate-item">
            {projectsData.map((project, index) => (
              <div 
                key={index}
                onClick={() => openProjectModal(project)}
                className="project-card card-hover-effect bg-gray-800 rounded-xl overflow-hidden cursor-pointer relative"
              >
                <div className={`h-48 bg-gradient-to-br ${project.imageGradient} relative`}>
                  <div className="absolute inset-0 flex items-center justify-center opacity-40">
                    <div className="w-16 h-16 bg-white rounded-full project-icon"></div>
                  </div>
                </div>
                <div className="p-6">
                  <div className="flex justify-between items-center mb-4">
                    <span className="text-xs text-gray-400">{project.year}</span>
                    <div className="flex gap-2">
                      {project.technologies.slice(0, 3).map((tech, i) => {
                        // Match the exact style of the programming skill tags but smaller
                        const techColors = {
                          'Python': { bg: 'bg-blue-500/20', text: 'text-blue-400' },
                          'React': { bg: 'bg-cyan-500/20', text: 'text-cyan-400' },
                          'TypeScript': { bg: 'bg-indigo-500/20', text: 'text-indigo-400' },
                          'Java': { bg: 'bg-yellow-500/20', text: 'text-yellow-400' },
                          'MATLAB': { bg: 'bg-green-500/20', text: 'text-green-400' },
                          'PyTorch': { bg: 'bg-purple-500/20', text: 'text-purple-400' },
                          'TensorFlow': { bg: 'bg-red-500/20', text: 'text-red-400' },
                          'AWS SageMaker': { bg: 'bg-orange-500/20', text: 'text-orange-400' },
                          'AWS Lambda': { bg: 'bg-orange-500/20', text: 'text-orange-400' },
                          'CloudWatch': { bg: 'bg-blue-500/20', text: 'text-blue-400' },
                          'AlexNet': { bg: 'bg-purple-500/20', text: 'text-purple-400' },
                          'Transfer Learning': { bg: 'bg-teal-500/20', text: 'text-teal-400' },
                          'Image Processing': { bg: 'bg-pink-500/20', text: 'text-pink-400' },
                          'Cryptography': { bg: 'bg-indigo-500/20', text: 'text-indigo-400' },
                          'Algorithm Design': { bg: 'bg-purple-500/20', text: 'text-purple-400' },
                          'Security': { bg: 'bg-red-500/20', text: 'text-red-400' }
                        };
                        
                        const { bg = 'bg-gray-500/20', text = 'text-gray-400' } = techColors[tech as keyof typeof techColors] || {};
                        
                        return (
                          <span key={i} className={`project-tag px-1.5 py-0.5 ${bg} ${text} rounded-lg text-[9px]`}>
                            {tech}
                          </span>
                        );
                      })}
                    </div>
                  </div>
                  <h3 className="text-xl font-bold mb-2">{project.title}</h3>
                  <p className="text-gray-400 text-sm mb-4 line-clamp-2">
                    {project.description}
                  </p>
                  <button className="text-blue-400 hover:text-blue-300 text-sm font-medium transition-colors flex items-center">
                    Learn More <ArrowRight size={14} className="ml-1" />
                  </button>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>
      
      {/* Blog Section - Fix visibility */}
      <section 
        id="blog" 
        className="py-20 animate-section bg-gray-900" 
        style={{opacity: 1, visibility: 'visible', display: 'block', position: 'relative', zIndex: 1}}
      >
        <div className="floating-element section-element-1" style={{background: 'rgba(96, 165, 250, 0.3)'}}></div>
        <div className="floating-element section-element-2" style={{background: 'rgba(192, 132, 252, 0.3)'}}></div>
        <div className="container mx-auto px-6">
          <h2 className="text-3xl font-bold mb-12 text-center animate-item">From My Blog</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 animate-item">
            {blogsData.map((blog, index) => (
              <div 
                key={index}
                onClick={() => openBlogModal(blog)}
                className="bg-gray-800 rounded-xl overflow-hidden cursor-pointer blog-card card-hover-effect"
              >
                <div className={`h-48 bg-gradient-to-br ${blog.imageGradient} relative`}>
                  <div className="absolute inset-0 flex items-center justify-center opacity-40">
                    <div className="w-16 h-16 bg-white rounded-full"></div>
                  </div>
                </div>
                <div className="p-6">
                  <div className="flex justify-between items-center mb-4">
                    <span className="text-xs text-gray-400">{blog.date}</span>
                    <span className="text-xs px-2 py-1 bg-gray-700 rounded-full text-gray-300">{blog.category}</span>
                  </div>
                  <h3 className="text-xl font-bold mb-2">{blog.title}</h3>
                  <p className="text-gray-400 text-sm mb-4">
                    {blog.description ?? "An exploration of advanced techniques and best practices..."}
                  </p>
                  <button 
                    className="text-blue-400 hover:text-blue-300 text-sm font-medium transition-colors flex items-center"
                  >
                    Read More <ArrowRight size={14} className="ml-1" />
                  </button>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>
      
      {/* Contact Section (Footer) */}
      <footer className="bg-gray-900 border-t border-gray-800 py-12">
        <div className="container mx-auto px-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8 items-center">
            <div>
              <h2 className="text-2xl font-bold mb-4">Let's Connect</h2>
              <p className="text-gray-400 mb-6">
                I'm currently open to AI/ML engineering opportunities. Feel free to reach out!
              </p>
              <div className="flex space-x-4">
                <a href="https://github.com/genjose12345" target="_blank" rel="noopener noreferrer" className="p-2 bg-gray-800 rounded-full text-gray-400 hover:text-white transition-colors">
                  <Github size={20} />
                </a>
                <a href="https://www.linkedin.com/in/jose-rodriguez-9a982b224/" target="_blank" rel="noopener noreferrer" className="p-2 bg-gray-800 rounded-full text-gray-400 hover:text-white transition-colors">
                  <Linkedin size={20} />
                </a>
                <button 
                  onClick={() => setIsContactModalOpen(true)}
                  className="p-2 bg-gray-800 rounded-full text-gray-400 hover:text-white transition-colors"
                >
                  <Mail size={20} />
                </button>
              </div>
            </div>
            <div className="text-right">
              <span className="text-lg font-bold text-blue-500">Jose Rodriguez</span>
              <p className="text-gray-500 mt-2">Machine Learning Engineer</p>
              <p className="text-gray-500">Dalton, GA</p>
            </div>
          </div>
          <div className="border-t border-gray-800 mt-8 pt-8 text-center text-sm text-gray-500">
            © {new Date().getFullYear()} Jose Rodriguez. All rights reserved.
          </div>
        </div>
      </footer>
      
      {/* Project Modal */}
      {isProjectModalOpen && projectModalData && (
        <div className="fixed inset-0 z-[90] flex items-center justify-center p-4">
          <div className="modal-backdrop absolute inset-0 bg-black bg-opacity-75" onClick={closeModals}></div>
          <div className="modal-content bg-gray-900 rounded-2xl overflow-hidden relative z-10 w-full max-w-4xl shadow-2xl">
            <div className={`gradient-header bg-gradient-to-r ${projectModalData.imageGradient} p-8 relative`}>
              <button 
                onClick={() => closeModals()}
                className="absolute top-4 right-4 bg-gray-900 bg-opacity-50 hover:bg-opacity-75 p-2 rounded-full transition-all hover:rotate-90 duration-300"
              >
                <X size={20} className="text-white" />
              </button>
              <h2 className="text-3xl font-bold mb-2">{projectModalData.title}</h2>
              <p className="text-xl text-gray-200 mb-4">{projectModalData.description}</p>
              <div className="flex flex-wrap gap-2 mt-4">
                {projectModalData.technologies.map((tech, index) => (
                  <span key={index} className="tech-tag px-3 py-1 bg-gray-800 bg-opacity-50 rounded-full text-sm text-gray-200">
                    {tech}
                  </span>
                ))}
              </div>
            </div>
            
            <div className="p-8 content-fade-in">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                <div className="space-y-6">
                  <div className="bg-gray-800 rounded-xl p-6 hover-card">
                    <h3 className="text-xl font-semibold mb-3 text-blue-400">Challenge</h3>
                    <p className="text-gray-300">{projectModalData.challenges}</p>
                  </div>
                  <div className="bg-gray-800 rounded-xl p-6 hover-card">
                    <h3 className="text-xl font-semibold mb-3 text-green-400">Solution</h3>
                    <p className="text-gray-300">{projectModalData.solution}</p>
                  </div>
                </div>
                <div className="bg-gray-800 rounded-xl p-6 hover-card">
                  <h3 className="text-xl font-semibold mb-3 text-purple-400">Results</h3>
                  <p className="text-gray-300">{projectModalData.results}</p>
                  <div className="mt-4 flex items-center justify-between text-sm text-gray-400">
                    <span className="flex items-center">
                      <Calendar size={16} className="mr-1" />
                      {projectModalData.year}
                    </span>
                    <button className="flex items-center text-blue-400 hover:text-blue-300 transition-colors">
                      View Demo <ExternalLink size={16} className="ml-1" />
                    </button>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
      
      {/* Blog Modal */}
      {isBlogModalOpen && blogModalData && (
        <div className="fixed inset-0 z-[90] flex items-center justify-center p-4">
          <div className="modal-backdrop absolute inset-0 bg-black bg-opacity-75" onClick={closeModals}></div>
          <div className="modal-content modal-highlight modal-glow bg-gray-900 rounded-2xl relative z-10 w-full max-w-4xl shadow-2xl blog-modal">
            <div className={`modal-header bg-gradient-to-r ${blogModalData.imageGradient} p-8 relative sticky top-0 z-10`}>
              <button 
                onClick={closeModals}
                className="absolute top-4 right-4 bg-gray-900 bg-opacity-50 hover:bg-opacity-75 p-2 rounded-full transition-all hover:rotate-90 duration-300"
              >
                <X size={20} className="text-white" />
              </button>
              <div className="flex items-center gap-4 mb-4">
                <span className="px-3 py-1 bg-gray-800 bg-opacity-50 rounded-full text-sm tech-tag">
                  {blogModalData.category}
                </span>
                <span className="text-gray-300 text-sm flex items-center tech-tag" style={{ '--tag-index': '1' } as React.CSSProperties}>
                  <Calendar size={16} className="mr-1" />
                  {blogModalData.date}
                </span>
              </div>
              <h2 className="text-3xl font-bold gradient-text">{blogModalData.title}</h2>
            </div>
            
            <div className="p-8 modal-content-wrapper">
              <div 
                className="prose prose-invert max-w-none blog-content"
                dangerouslySetInnerHTML={{ __html: blogModalData.content }}
              ></div>
              
              <div className="mt-8 pt-6 border-t border-gray-800">
                <div className="flex justify-between items-center">
                  <button className="text-blue-400 hover:text-blue-300 transition-colors flex items-center hover-card">
                    Share Post <ExternalLink size={16} className="ml-1" />
                  </button>
                  <div className="flex gap-2">
                    {['#tech', '#development', '#coding'].map((tag, index) => (
                      <span 
                        key={index} 
                        className="tech-tag text-sm text-gray-400 hover:text-blue-400 transition-colors cursor-pointer"
                        style={{ '--tag-index': index }}
                      >
                        {tag}
                      </span>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
      
      {/* Contact Modal */}
      {isContactModalOpen && (
        <div className="fixed inset-0 z-[90] flex items-center justify-center p-4">
          <div className="modal-backdrop absolute inset-0 bg-black bg-opacity-75 backdrop-blur-sm" onClick={closeModals}></div>
          <div className="modal-content modal-highlight modal-glow modal-open bg-gray-900 rounded-2xl overflow-hidden relative z-10 w-full max-w-md shadow-2xl">
            <div className="modal-header bg-gradient-to-r from-blue-600 via-indigo-600 to-blue-600 p-6 relative">
              <button 
                onClick={closeModals}
                className="absolute top-4 right-4 bg-gray-900 bg-opacity-50 hover:bg-opacity-75 p-2 rounded-full transition-all hover:rotate-90 duration-300"
              >
                <X size={20} className="text-white" />
              </button>
              <h2 className="text-2xl font-bold mb-1">Get In Touch</h2>
              <p className="text-gray-200 text-sm opacity-80">I'd love to hear from you about opportunities, collaborations, or just to connect!</p>
            </div>
            
            <div className="p-6 space-y-4 content-fade-in">
              <div className="bg-gray-800/50 hover:bg-gray-800/70 transition-colors rounded-xl p-5 hover-card">
                <form className="space-y-4">
                  <div>
                    <label htmlFor="name" className="block text-sm font-medium text-gray-400 mb-1">Name</label>
                    <input 
                      type="text" 
                      id="name"
                      className="w-full bg-gray-700/50 border border-gray-600 rounded-lg px-4 py-2 text-white focus:outline-none focus:border-blue-500"
                      placeholder="Your name"
                    />
                  </div>
                  <div>
                    <label htmlFor="email" className="block text-sm font-medium text-gray-400 mb-1">Email</label>
                    <input 
                      type="email" 
                      id="email"
                      className="w-full bg-gray-700/50 border border-gray-600 rounded-lg px-4 py-2 text-white focus:outline-none focus:border-blue-500"
                      placeholder="your.email@example.com"
                    />
                  </div>
                  <div>
                    <label htmlFor="subject" className="block text-sm font-medium text-gray-400 mb-1">Subject</label>
                    <input 
                      type="text" 
                      id="subject"
                      className="w-full bg-gray-700/50 border border-gray-600 rounded-lg px-4 py-2 text-white focus:outline-none focus:border-blue-500"
                      placeholder="What's this about?"
                    />
                  </div>
                  <div>
                    <label htmlFor="message" className="block text-sm font-medium text-gray-400 mb-1">Message</label>
                    <textarea 
                      id="message"
                      rows={4}
                      className="w-full bg-gray-700/50 border border-gray-600 rounded-lg px-4 py-2 text-white focus:outline-none focus:border-blue-500 resize-none"
                      placeholder="Your message here..."
                    ></textarea>
                  </div>
                  <button 
                    type="submit"
                    className="w-full bg-blue-600 hover:bg-blue-700 text-white rounded-lg px-4 py-3 font-medium transition-colors"
                  >
                    Send Message
                  </button>
                </form>
              </div>
              
              <div className="bg-gray-800/50 hover:bg-gray-800/70 transition-colors rounded-xl p-5 hover-card">
                <h4 className="text-lg font-semibold mb-3 text-blue-400 flex items-center">
                  <Mail className="mr-2" /> Direct Contact
                </h4>
                <p className="text-gray-300">
                  You can also email me directly at:<br/>
                  <a href="mailto:genjose1231@gmail.com" className="text-blue-400 hover:text-blue-300 transition-colors">
                    genjose1231@gmail.com
                  </a>
                </p>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Timeline Modal */}
      {isTimelineModalOpen && timelineModalData && (
        <div className="fixed inset-0 z-[90] flex items-center justify-center p-4">
          <div className="absolute inset-0 bg-black bg-opacity-75 backdrop-blur-sm" onClick={closeModals}></div>
          <div className="modal-content bg-gray-900 rounded-2xl overflow-hidden relative z-10 w-full max-w-3xl shadow-2xl">
            <div className={`modal-header bg-gradient-to-r from-${timelineModalData.color}-600 via-${timelineModalData.color}-500 to-${timelineModalData.color}-600 p-6 relative sticky top-0 z-10`}>
              <button 
                onClick={closeModals}
                className="absolute top-4 right-4 bg-gray-900 bg-opacity-50 hover:bg-opacity-75 p-2 rounded-full transition-all hover:rotate-90 duration-300"
              >
                <X size={20} className="text-white" />
              </button>
              <h2 className="text-2xl font-bold gradient-text">{timelineModalData.title}</h2>
              <p className="text-lg text-gray-200">{timelineModalData.company}</p>
              <div className="flex justify-between text-gray-200 text-sm mt-2">
                <span className="flex items-center">
                  <MapPin size={16} className="mr-1" />
                  {timelineModalData.location}
                </span>
                <span className="flex items-center">
                  <Calendar size={16} className="mr-1" />
                  {timelineModalData.date}
                </span>
              </div>
            </div>
            
            <div className="p-6 space-y-6 max-h-[70vh] overflow-y-auto content-fade-in">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="bg-gray-800/50 hover:bg-gray-800/70 transition-colors rounded-xl p-5 hover-card">
                  <h4 className="text-lg font-semibold mb-3 text-blue-400 flex items-center">
                    <Code className="mr-2" /> Technologies
                  </h4>
                  <div className="flex flex-wrap gap-2">
                    {timelineModalData.technologies?.map((tech, index) => (
                      <span key={index} className="px-3 py-1 bg-gray-700/50 hover:bg-gray-700/70 rounded-full text-sm text-gray-300 transition-colors">
                        {tech}
                      </span>
                    ))}
                  </div>
                </div>
                
                <div className="bg-gray-800/50 hover:bg-gray-800/70 transition-colors rounded-xl p-5 hover-card">
                  <h4 className="text-lg font-semibold mb-3 text-green-400 flex items-center">
                    <Award className="mr-2" /> Skills
                  </h4>
                  <div className="flex flex-wrap gap-2">
                    {timelineModalData.skills?.map((skill, index) => (
                      <span key={index} className="px-3 py-1 bg-gray-700/50 hover:bg-gray-700/70 rounded-full text-sm text-gray-300 transition-colors">
                        {skill}
                      </span>
                    ))}
                  </div>
                </div>
                
                {timelineModalData.impact && (
                  <div className="bg-gray-800/50 hover:bg-gray-800/70 transition-colors rounded-xl p-5 hover-card">
                    <h4 className="text-lg font-semibold mb-3 text-purple-400 flex items-center">
                      <ChevronDown className="mr-2" /> Impact
                    </h4>
                    <p className="text-gray-300">{timelineModalData.impact}</p>
                  </div>
                )}
                
                {timelineModalData.tools && (
                  <div className="bg-gray-800/50 hover:bg-gray-800/70 transition-colors rounded-xl p-5 hover-card">
                    <h4 className="text-lg font-semibold mb-3 text-yellow-400 flex items-center">
                      <Server className="mr-2" /> Tools
                    </h4>
                    <div className="flex flex-wrap gap-2">
                      {timelineModalData.tools?.map((tool, index) => (
                        <span key={index} className="px-3 py-1 bg-gray-700/50 hover:bg-gray-700/70 rounded-full text-sm text-gray-300 transition-colors">
                          {tool}
                        </span>
                      ))}
                    </div>
                  </div>
                )}
              </div>

              <div className="bg-gray-800/50 hover:bg-gray-800/70 transition-colors rounded-xl p-5 hover-card">
                <h4 className="text-lg font-semibold mb-3 text-orange-400">Key Achievements</h4>
                <ul className="space-y-3">
                  {timelineModalData.details.map((detail, index) => (
                    <li key={index} className="flex items-start p-2 rounded-lg hover:bg-gray-700/50 transition-colors">
                      <span className="text-orange-400 mr-2">•</span>
                      <span className="text-gray-300">{detail}</span>
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          </div>
        </div>
      )}
    </Layout>
  );
};

export default Portfolio;
