import React from 'react';
import Card from './Card';
import './Cardcomponents.css';
import login from './image.png';
import graphs from './graphs.jpg';
import cyber from './cyber_research.jpg';

const CardComponents = () => {
    const cardsData = [
        {
            id: 1,
            title: "Chess tournament",
            content: "This project is about creating a chess tournament system That schools can use to auto generate matches along with add remove  players and view the tournament standings.",
            image: login
        },
        {
            id: 2,
            title: "Transfer Learning AI",
            content: "In this project I used Transfer Learning to detect diabetic retinophaty that can be found in the eyes.",
            image: graphs
        },
        {
            id: 3,
            title: "Undergrduate Research",
            content: "This project I did undergraduate research in order to create a decryption code that will require a key to decrpty a message and if no key is given then you will get an approximation for the size  of the key.",

            image: cyber
        }
    ];

    return (
        <div className="card-container">
            {cardsData.map((card) => (
                <Card
                    key={card.id}
                    title={card.title}
                    content={card.content}
                    image={card.image}
                />
            ))}
        </div>
    );
};
export default CardComponents;