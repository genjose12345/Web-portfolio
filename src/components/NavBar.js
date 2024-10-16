// Import React library
import React from 'react';
// Import CSS file for NavBar styling
import './NavBar.css';

// Define NavBar functional component
function NavBar() {
    // Define a function to handle smooth scrolling to a section
    const scrollToSection = (sectionId) => {
        // Find the element with the given sectionId
        const element = document.getElementById(sectionId);
        // If the element exists, scroll to it smoothly
        if (element) {
            element.scrollIntoView({ behavior: 'smooth' });
        }
    };

    // Return JSX for the NavBar component
    return (
        // Main navigation container
        <nav className="navbar">
            {/* Logo container */}
            <div className="navbar-logo">
                {/* Logo link that scrolls to home when clicked */}
                <a href="#home" onClick={() => scrollToSection('home')}>Shitzu</a>
            </div>
            {/* Navigation links container */}
            <ul className="navbar-links">
                {/* Home link */}
                <li><a href="#home" onClick={() => scrollToSection('home')}>Home</a></li>
                {/* About link */}
                <li><a href="#about" onClick={() => scrollToSection('about')}>About</a></li>
                {/* Skills link */}
                <li><a href="#skills" onClick={() => scrollToSection('skills')}>Skills</a></li>
                {/* Projects link */}
                <li><a href="#projects" onClick={() => scrollToSection('projects')}>Projects</a></li>
                {/* Contact link */}
                <li><a href="#contact" onClick={() => scrollToSection('contact')}>Contact</a></li>
            </ul>
        </nav>
    );
}

// Export the NavBar component for use in other parts of the application
export default NavBar;