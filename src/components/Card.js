import React from 'react';
import './Card.css';

function Card({ title, content, image }) {
    return (
        <div className="card">
            <img src={image} alt={title} className="card-image" />
            <div className="card-content">
                <h5 className="card-title">{title}</h5>
                <p className="card-text">{content}</p>
            </div>
        </div>
    );
}

export default Card;