import React, { useEffect } from 'react';
import './App.css';
import NavBar from './components/NavBar';
import CardComponents from './components/Cardcomponents';
import { initSkillsAnimation } from './SkillsAnimation';
import ScrollButton from './components/ScrollButton';  // Import the ScrollButton component

//https://demo.wpoperation.com/arrival-me/
function App() {
  useEffect(() => {
    // Initialize the skills animation
    initSkillsAnimation();

    // Dynamically load the Font Awesome script
    const fontAwesomeScript = document.createElement('script');
    fontAwesomeScript.src = 'https://kit.fontawesome.com/1797ec04b3.js';
    fontAwesomeScript.crossOrigin = 'anonymous';
    document.body.appendChild(fontAwesomeScript);

    return () => {
      // Clean up the script when the component unmounts
      document.body.removeChild(fontAwesomeScript);
    };
  }, []);

  return (
    <div className="app">
      <header className="header">
        <NavBar />
      </header>
      <main className="main-content">
        <section id="home" className="hero-section">
          <p><b className="hero-name">I'm Jose Rodriguez</b></p>
          <p><b className="hero-job">IS Developer</b></p>
          <p className="hero-description">I'm currently an IS/Developer creating software to be deployed overseas with a passion in AI and Machine Learning</p>
        </section>
        <section id="about" className="about-section">
          <div className="about-content">
            <h1><u>About Me</u></h1>
            <p>Education:Dalton State College Associates of Science in Computer Science, Kennesaw State University Bachelor of Science in Computer Science</p>
            <p>Goals: getting my Master's and PHD</p>
            <p>Who am I? I'm a first generation college student looking to make change in the world with the advancements of technology.Some of my hobbie include working out and learning new programming languages</p>
          </div>
        </section>
        <section id="skills" className="skill-section">
          <div className="skill-title">
            <h1>My Skills</h1>
          </div>
          <div className="skills-container">
            <div className="skills-column">
              <div className="skill-item">
                <p>JAVA</p>
                <div className="container">
                  <div className="skill java"></div>
                </div>
              </div>
              <div className="skill-item">
                <p>MATLAB</p>
                <div className="container">
                  <div className="skill matlab"></div>
                </div>
              </div>
              <div className="skill-item">
                <p>R</p>
                <div className="container">
                  <div className="skill r"></div>
                </div>
              </div>
            </div>
            <div className="skills-column">
              <div className="skill-item">
                <p>HTML</p>
                <div className="container">
                  <div className="skill html"></div>
                </div>
              </div>
              <div className="skill-item">
                <p>CSS</p>
                <div className="container">
                  <div className="skill css"></div>
                </div>
              </div>
              <div className="skill-item">
                <p>JAVASCRIPT</p>
                <div className="container">
                  <div className="skill javascript"></div>
                </div>
              </div>
            </div>
          </div>
        </section>
        <section id="projects" className="projects-section">
          <div className="projects-title">
            <h1>Projects</h1>
          </div>
          <CardComponents />
        </section>
        <section id="contact" className="contact-section">
          <h1>Contact Me</h1>
          <div className="contact-links">
            <p> <i class="fa-solid fa-phone"></i>Text:  </p>
            <p> 706-618-1178</p>
            <p> <i class="fa-regular fa-envelope"></i> Email: </p>
            <p> genjose1231@gmail.com</p>
            <p> <i class="fa-brands fa-linkedin"></i> Linkedin:</p>
            <p> <a href="https://www.linkedin.com/in/jose-rodriguez-9a982b224"> https://www.linkedin.com/in/jose-rodriguez-9a982b224</a></p>
            <p> <i class="fa-brands fa-github"></i> GitHub: </p>
            <p><a href="https://github.com/genjose12345"> https://github.com/genjose12345</a></p>
          </div>
        </section>
      </main>
      <ScrollButton />
    </div>
  );
}

export default App;