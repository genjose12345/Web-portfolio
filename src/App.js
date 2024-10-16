import React from 'react';
import './App.css';
import NavBar from './components/NavBar';
import CardComponents from './components/Cardcomponents';
//https://demo.wpoperation.com/arrival/demos/
function App() {
  return (
    <div className="app">
      <header className="header">
        <NavBar />
      </header>
      <main className="main-content">
        <section className="hero-section">
          <p><b className="hero-name">I'm Jose Rodriguez</b></p>
          <p><b className="hero-job">IS Developer</b></p>
          <p className="hero-description">I'm currently an IS/Developer creating software to be deployed overseas with a passion in AI and Machine Learning</p>
        </section>
        <section className="about-section">
          <h1><u>About Me</u></h1>
          <p>Education: Associates of Science in Computer Science, Bachelor of Science in Computer Science</p>
          <p>Where: Dalton State College, Kennesaw State University</p>
          <p>Goals: getting my Master's and PHD</p>
          <p>Who am i? I'm a first generation college student looking to make change in the world with the advancements of technology</p>
        </section>
        <section class="skill-section">
          <div class="skill-title">
            <h1>My Skills</h1>
            <p>a list of some of my skills</p>
            <div class="skills-container">
              <div class="skills-column">
                <div class="skill-item">
                  <p>JAVA</p>
                  <div class="container">
                    <div class="skill java">80%</div>
                  </div>
                </div>
                <div class="skill-item">
                  <p>MATLAB</p>
                  <div class="container">
                    <div class="skill matlab">60%</div>
                  </div>
                </div>
                <div class="skill-item">
                  <p>R</p>
                  <div class="container">
                    <div class="skill r">60%</div>
                  </div>
                </div>
              </div>
              <div class="skills-column">
                <div class="skill-item">
                  <p>HTML</p>
                  <div class="container">
                    <div class="skill html">50%</div>
                  </div>
                </div>
                <div class="skill-item">
                  <p>CSS</p>
                  <div class="container">
                    <div class="skill css">50%</div>
                  </div>
                </div>
                <div class="skill-item">
                  <p>JAVASCRIPT</p>
                  <div class="container">
                    <div class="skill javascript">50%</div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>
        <section className="projects-section">
          <div className="projects-title">
            <h1>Projects</h1>
          </div>
          <CardComponents />
        </section>
        <section className="contact-section">
          <h1>Contact Me</h1>
          <div className="contact-links">
            <p>My <a href="https://www.linkedin.com/in/jose-rodriguez-9a982b224">LinkedIn</a></p>
            <p>My <a href="https://github.com/genjose12345">GitHub</a></p>
          </div>
        </section>
      </main>
    </div>
  );
}

export default App;