// NavBar.js
import React, { useState } from 'react';
import './NavBar.css';

function NavBar() {
    const [showSecret, setShowSecret] = useState(false);

    const scrollToSection = (sectionId) => {
        const element = document.getElementById(sectionId);
        if (element) {
            element.scrollIntoView({ behavior: 'smooth' });
        }
    };

    const toggleSecret = () => {
        setShowSecret(!showSecret);
    };

    return (
        <nav className="navbar">
            <div className="navbar-logo">
                <a href="#logo" onClick={() => { scrollToSection(); toggleSecret(); }}>Shitzu</a>
            </div>
            <div className="navbar-center">
                {showSecret && (
                    <>
                        <i className="fa-solid fa-person-cane"></i>
                        <p>You found grandpa dave congrats :)</p>
                    </>
                )}
            </div>
            <ul className="navbar-links">
                <li><a href="#home" onClick={() => scrollToSection('home')}>Home</a></li>
                <li><a href="#about" onClick={() => scrollToSection('about')}>About</a></li>
                <li><a href="#skills" onClick={() => scrollToSection('skills')}>Skills</a></li>
                <li><a href="#projects" onClick={() => scrollToSection('projects')}>Projects</a></li>
                <li><a href="#contact" onClick={() => scrollToSection('contact')}>Contact</a></li>
            </ul>
        </nav>
    );
}

export default NavBar;