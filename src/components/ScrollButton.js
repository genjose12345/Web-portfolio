import React, { useState, useEffect } from "react";
import styled from "styled-components";

const Button = styled.div`
  position: fixed;
  right: 20px;
  bottom: 20px;
  height: 70px;  // Further reduced height
  width: 35px;   // Further reduced width
  background-color: #ffd700;
  border-radius: 17.5px;  // Half of the width for oval shape
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  cursor: pointer;
  transition: all 0.3s ease;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);

  &:hover {
    background-color: #ffc700;
  }
`;

const Arrows = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  transition: transform 0.3s ease;

  ${Button}:hover & {
    transform: translateY(-10px);  // Slightly reduced movement on hover
  }
`;

const Arrow = styled.div`
  width: 8px;   // Smaller arrow
  height: 8px;  // Smaller arrow
  border-left: 2px solid white;
  border-top: 2px solid white;
  transform: rotate(45deg);
  margin: -2px 0;  // Reduced margin to bring arrows closer
`;

const ScrollButton = () => {
    const [visible, setVisible] = useState(false);

    const toggleVisible = () => {
        const scrolled = document.documentElement.scrollTop;
        if (scrolled > 300) {
            setVisible(true);
        } else if (scrolled <= 300) {
            setVisible(false);
        }
    };

    const scrollToTop = () => {
        window.scrollTo({
            top: 0,
            behavior: "smooth"
        });
    };

    useEffect(() => {
        window.addEventListener("scroll", toggleVisible);
        return () => window.removeEventListener("scroll", toggleVisible);
    }, []);

    return (
        <Button onClick={scrollToTop} style={{ display: visible ? 'flex' : 'none' }}>
            <Arrows>
                <Arrow />
                <Arrow />
            </Arrows>
        </Button>
    );
};

export default ScrollButton;