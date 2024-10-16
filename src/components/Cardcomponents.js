import React from 'react';
import Card from './Card';
import './Cardcomponents.css';

const CardComponents = () => {
    const cardsData = [
        {
            id: 1,
            title: "Chess tournament",
            content: "This project is about creating a chess tournament system That schools can use to auto generate matches along with add remove  players and view the tournament standings.",
            imageUrl: "test1.jpg"
        },
        {
            id: 2,
            title: "Deep learning AI",
            content: "This project i used deep learning to detect diabetic retinophaty that can be found in the eyes.",
            imageUrl: "test2.jpg"
        },
        {
            id: 3,
            title: "Undergrduate Research",
            content: "This project I did undergraduate research in order to create a decryption code that will require a key to decrpty a message and if no key is given then will give you the most likly size of the key",
            imageUrl: "test3.jpg"
        }
    ];

    return (
        <div className="card-container">
            {cardsData.map((card) => (
                <Card
                    key={card.id}
                    title={card.title}
                    content={card.content}
                    imageUrl={card.imageUrl}
                />
            ))}
        </div>
    );
};
export default CardComponents;