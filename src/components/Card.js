import React from 'react';
import './Card.css';

function Card({ title, content, imageUrl }) {
    return (
        <div className="card">
            <img src={imageUrl} alt={title} className="card-image" />
            <div className="card-content">
                <h5 className="card-title">{title}</h5>
                <p className="card-text">{content}</p>
            </div>
        </div>
    );
}

export default Card;