import React, { useState, useEffect } from 'react';

const Firework: React.FC = () => {
  const style = {
    left: `${Math.random() * 100}%`,
    top: `${Math.random() * 100}%`,
    animationDelay: `${Math.random() * 2}s`,
    '--hue': `${Math.random() * 360}deg`,
    '--scale': `${0.5 + Math.random() * 0.5}`,
  } as React.CSSProperties;

  return (
    <div className="firework-container" style={style}>
      <div className="firework-base"></div>
      {Array.from({ length: 24 }).map((_, i) => (
        <div 
          key={i} 
          className="firework-particle" 
          style={{ 
            '--angle': `${(i * 15)}deg`,
            '--delay': `${Math.random() * 0.2}s`,
            '--distance': `${50 + Math.random() * 50}px`
          } as React.CSSProperties}
        ></div>
      ))}
    </div>
  );
};

export const ParticleBackground: React.FC = () => {
  const [fireworks, setFireworks] = useState<number[]>([]);

  useEffect(() => {
    const interval = setInterval(() => {
      setFireworks(prev => [...prev, Date.now()].slice(-8));
    }, 1500);

    return () => clearInterval(interval);
  }, []);

  return (
    <div className="fixed inset-0 pointer-events-none overflow-hidden">
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_center,rgba(37,99,235,0.15)_0,rgba(0,0,0,0)_70%)]"></div>
      {fireworks.map(key => (
        <Firework key={key} />
      ))}
      
      {/* Floating particles */}
      {Array.from({ length: 30 }).map((_, i) => (
        <div
          key={`particle-${i}`}
          className="absolute rounded-full bg-blue-500 opacity-10 floating"
          style={{
            width: Math.random() * 10 + 5 + 'px',
            height: Math.random() * 10 + 5 + 'px',
            left: Math.random() * 100 + '%',
            top: Math.random() * 100 + '%',
            animationDelay: Math.random() * 2 + 's',
            animationDuration: Math.random() * 3 + 3 + 's'
          }}
        ></div>
      ))}
      
      {/* Gradient lines */}
      <div className="absolute inset-0 opacity-20">
        {Array.from({ length: 3 }).map((_, i) => (
          <div
            key={`line-${i}`}
            className="absolute h-px w-full gradient-line"
            style={{
              top: (i + 1) * 25 + '%',
              animationDelay: i * 0.5 + 's'
            }}
          ></div>
        ))}
      </div>
    </div>
  );
}; 