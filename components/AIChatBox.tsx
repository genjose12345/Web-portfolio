import React, { useState, useEffect, useRef } from 'react';
import { MessageCircle, X, Minimize, Maximize, Send, RefreshCw } from 'lucide-react';

type Message = {
  sender: 'user' | 'ai';
  text: string;
  isGame?: boolean;
  gameComponent?: React.ReactNode;
};

type GameState = {
  type: 'none' | 'rps' | 'tictactoe' | 'pong';
  data: any;
};

// Add this interface before the AIChatBox component
interface TicTacToeProps {
  gameState: GameState;
  setGameState: React.Dispatch<React.SetStateAction<GameState>>;
  setMessages: React.Dispatch<React.SetStateAction<Message[]>>;
}

// Add this interface
interface PongProps {
  gameState: GameState;
  setGameState: React.Dispatch<React.SetStateAction<GameState>>;
  setMessages: React.Dispatch<React.SetStateAction<Message[]>>;
}

// Add these helper functions before the TicTacToeGame component
const calculateWinner = (squares: Array<string | null>): string | null => {
  const lines = [
    [0, 1, 2],
    [3, 4, 5],
    [6, 7, 8],
    [0, 3, 6],
    [1, 4, 7],
    [2, 5, 8],
    [0, 4, 8],
    [2, 4, 6],
  ];
  
  for (let i = 0; i < lines.length; i++) {
    const [a, b, c] = lines[i];
    if (squares[a] && squares[a] === squares[b] && squares[a] === squares[c]) {
      return squares[a];
    }
  }
  return null;
};

const getAIMove = (board: Array<string | null>): number => {
  // Try to win
  for (let i = 0; i < 9; i++) {
    if (!board[i]) {
      const testBoard = [...board];
      testBoard[i] = 'O';
      if (calculateWinner(testBoard) === 'O') return i;
    }
  }

  // Block player win
  for (let i = 0; i < 9; i++) {
    if (!board[i]) {
      const testBoard = [...board];
      testBoard[i] = 'X';
      if (calculateWinner(testBoard) === 'X') return i;
    }
  }

  // Take center
  if (!board[4]) return 4;

  // Take corners
  const corners = [0, 2, 6, 8];
  const availableCorners = corners.filter(i => !board[i]);
  if (availableCorners.length > 0) {
    return availableCorners[Math.floor(Math.random() * availableCorners.length)];
  }

  // Take any available space
  const available = board.map((square, i) => square ? -1 : i).filter(i => i !== -1);
  return available[Math.floor(Math.random() * available.length)];
};

// Add this component before the AIChatBox component
const TicTacToeGame: React.FC<TicTacToeProps> = ({ gameState, setGameState, setMessages }) => {
  const renderSquare = (index: number) => {
    const board = gameState.data.board;
    return (
      <button
        className="w-14 h-14 bg-gray-700 border border-gray-600 flex items-center justify-center text-2xl font-bold text-white hover:bg-gray-600 transition-colors"
        onClick={() => handleClick(index)}
        disabled={board[index] !== null || !gameState.data.playerTurn}
      >
        {board[index]}
      </button>
    );
  };

  const handleClick = (index: number) => {
    if (gameState.data.board[index] || gameState.data.gameOver) return;

    // Player move
    const newBoard = [...gameState.data.board];
    newBoard[index] = 'X';
    
    // Check for winner
    const winner = calculateWinner(newBoard);
    if (winner) {
      setGameState({
        ...gameState,
        data: { ...gameState.data, board: newBoard, gameOver: true, winner }
      });
      return;
    }

    // AI move
    setTimeout(() => {
      const aiMove = getAIMove(newBoard);
      if (aiMove !== -1) {
        newBoard[aiMove] = 'O';
        const winner = calculateWinner(newBoard);
        setGameState({
          ...gameState,
          data: {
            board: newBoard,
            playerTurn: true,
            gameOver: !!winner,
            winner: winner || null
          }
        });
      }
    }, 500);
  };

  return (
    <div className="p-4 bg-gray-800 rounded-lg">
      <div className="grid grid-cols-3 gap-2 mb-4">
        {[0, 1, 2, 3, 4, 5, 6, 7, 8].map(index => renderSquare(index))}
      </div>
      <div className="text-center text-gray-300">
        {gameState.data.gameOver 
          ? `Game Over! ${gameState.data.winner === 'X' ? 'You won!' : 'AI won!'}`
          : `${gameState.data.playerTurn ? 'Your turn' : 'AI thinking...'}`}
      </div>
    </div>
  );
};

// Add the Pong component
const PongGame: React.FC<PongProps> = ({ gameState, setGameState, setMessages }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [gameLoop, setGameLoop] = useState<number | null>(null);
  const [paddle, setPaddle] = useState({ x: 175, y: 280, width: 50, height: 10 });
  const [ball, setBall] = useState({ x: 200, y: 150, dx: 2, dy: 3, radius: 5 });
  const [aiPaddle, setAiPaddle] = useState({ x: 175, y: 10, width: 50, height: 10 });

  // Add game logic here (from ai-chatbot-component.txt)
  // Including ball movement, collision detection, AI paddle movement, etc.

  return (
    <div className="p-4 bg-gray-800 rounded-lg">
      <canvas
        ref={canvasRef}
        width={400}
        height={300}
        className="bg-gray-900 border border-gray-700 rounded-lg"
        onMouseMove={(e) => {
          const rect = canvasRef.current?.getBoundingClientRect();
          if (rect) {
            const mouseX = e.clientX - rect.left;
            setPaddle(prev => ({
              ...prev,
              x: Math.max(0, Math.min(350, mouseX - 25))
            }));
          }
        }}
      />
      <div className="mt-2 text-center text-gray-300">
        Move your mouse to control the paddle
      </div>
    </div>
  );
};

const AIChatBox: React.FC = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [isMinimized, setIsMinimized] = useState(false);
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [gameState, setGameState] = useState<GameState>({ type: 'none', data: null });
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Cool facts about AI and technology
  const coolFacts = [
    "The term 'Artificial Intelligence' was first coined in 1956 at the Dartmouth Conference.",
    "The first neural network computer was called the SNARC and was built in 1951.",
    "AI algorithms can now compose music that is indistinguishable from pieces written by human composers.",
    "Facial recognition AI can identify individuals even with masks on, with up to 95% accuracy.",
    "AI in self-driving cars processes over 1GB of data per second.",
    "AI can predict heart attacks and strokes more accurately than standard medical methods.",
    "Deep Blue was the first computer to beat a world chess champion, Garry Kasparov, in 1997.",
    "Machine learning models can now generate realistic images of people who don't exist.",
    "AI assistants like Siri and Alexa use natural language processing to understand human speech.",
    "AlphaGo defeated the world champion at Go, a game with more possible positions than atoms in the universe.",
    "AI systems are being used to help discover new drugs by predicting compound behavior.",
    "Recommendation systems like those used by Netflix use AI to suggest content.",
    "Some modern AI models have over 175 billion parameters.",
    "AI-powered drones are being used for delivery services and search and rescue.",
    "Machine learning can detect cancer from medical scans with higher accuracy than some doctors.",
    "AI language models can write code, stories, and essays based on simple prompts.",
    "Reinforcement learning helps AI learn by trial and error, similar to humans.",
    "Computer vision can identify objects in images with greater than 99% accuracy.",
    "AI can now generate photorealistic faces indistinguishable from real photos.",
    "Quantum computing could exponentially increase AI capabilities in the future."
  ];

  // Initial greeting when chat is first opened
  useEffect(() => {
    if (isOpen && messages.length === 0) {
      setTimeout(() => {
        setMessages([
          { 
            sender: 'ai', 
            text: "Hi there! I'm Jose's AI assistant. I can share random facts about AI or we can play some games! Try asking me for a 'fun fact' or type 'play games' to see what games we can play!" 
          }
        ]);
      }, 500);
    }
  }, [isOpen, messages.length]);

  // Auto-scroll to bottom of messages
  useEffect(() => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages]);

  // Handle toggle chat open/closed
  const toggleChat = () => {
    setIsOpen(!isOpen);
    setIsMinimized(false);
  };

  // Handle toggle minimize/maximize
  const toggleMinimize = () => {
    setIsMinimized(!isMinimized);
  };

  // Get a random fact
  const getRandomFact = () => {
    const randomIndex = Math.floor(Math.random() * coolFacts.length);
    return coolFacts[randomIndex];
  };

  // Process user input and generate response
  const handleSendMessage = () => {
    if (inputValue.trim() === '') return;
    
    const userMessage: Message = { sender: 'user', text: inputValue };
    setMessages(prev => [...prev, userMessage]);
    
    const userInput = inputValue.toLowerCase();
    setInputValue('');
    
    setTimeout(() => {
      let aiResponse: Message = { sender: 'ai', text: '' };
      
      if (gameState.type !== 'none') {
        if (userInput === 'quit' || userInput === 'exit') {
          setGameState({ type: 'none', data: null });
          aiResponse.text = "Game ended! What would you like to do next?";
          setMessages(prev => [...prev, aiResponse]);
        }
        return;
      }
      
      if (userInput.includes('play tic') || userInput.includes('tictactoe')) {
        startTicTacToe();
        return;
      } else if (userInput.includes('play pong')) {
        startPong();
        return;
      } else if (userInput.includes('play rps')) {
        startRockPaperScissors();
        return;
      } else if (userInput.includes('fact')) {
        aiResponse.text = getRandomFact();
      } else if (userInput.includes('play') || userInput.includes('game')) {
        aiResponse.text = "I can play Rock Paper Scissors, Tic-tac-toe, or Pong! Type 'play rps', 'play tic-tac-toe', or 'play pong' to start a game.";
      } else if (userInput.includes('hello') || userInput.includes('hi')) {
        aiResponse.text = "Hello! How can I help you today?";
      } else if (userInput.includes('who are you')) {
        aiResponse.text = "I'm an AI assistant created to showcase Jose's programming skills! I can share interesting facts and play games.";
      } else if (userInput.includes('help')) {
        aiResponse.text = "I can share random facts about AI (just ask for a 'fact') or we can play games! Try 'play rps', 'play tic-tac-toe', or 'play pong'.";
      } else {
        aiResponse.text = "I'm not sure how to respond to that. Try asking for a 'fact' or type 'play games' to see what games we can play!";
      }
      
      setMessages(prev => [...prev, aiResponse]);
    }, 500);
  };

  // Handle Enter key in input
  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleSendMessage();
    }
  };

  // Add this function to start the Tic-tac-toe game
  const startTicTacToe = () => {
    const initialState = {
      type: 'tictactoe' as const,
      data: {
        board: Array(9).fill(null),
        playerTurn: true,
        gameOver: false,
        winner: null
      }
    };
    
    setGameState(initialState);
    
    const aiResponse: Message = { 
      sender: 'ai', 
      text: "Let's play Tic-tac-toe! You're X, I'm O. Click on a square to make your move. Type 'quit' to exit.",
      isGame: true,
      gameComponent: <TicTacToeGame 
        gameState={initialState} 
        setGameState={setGameState} 
        setMessages={setMessages} 
      />
    };
    
    setMessages(prev => [...prev, aiResponse]);
  };

  // Add the startPong function
  const startPong = () => {
    setGameState({
      type: 'pong',
      data: {
        playerScore: 0,
        aiScore: 0,
        gameStarted: false
      }
    });

    const aiResponse: Message = {
      sender: 'ai',
      text: "Let's play Pong! Move your mouse to control the paddle. First to 5 points wins!",
      isGame: true,
      gameComponent: <PongGame gameState={gameState} setGameState={setGameState} setMessages={setMessages} />
    };

    setMessages(prev => [...prev, aiResponse]);
  };

  return (
    <div className="fixed bottom-6 right-6 z-[100]">
      {/* Chat window */}
      {isOpen && (
        <div className={`bg-gray-900 border border-gray-700 rounded-lg shadow-xl mb-4 transition-all duration-300 overflow-hidden flex flex-col ${
          isMinimized ? 'w-72 h-12' : 'w-80 sm:w-96 h-96'
        }`}>
          {/* Chat header */}
          <div className="bg-gray-800 p-3 flex justify-between items-center border-b border-gray-700">
            <div className="flex items-center space-x-2">
              <div className="w-2 h-2 rounded-full bg-green-500"></div>
              <h3 className="font-medium text-white">AI Assistant</h3>
            </div>
            <div className="flex items-center space-x-2">
              <button 
                onClick={toggleMinimize}
                className="text-gray-400 hover:text-white transition-colors"
              >
                {isMinimized ? <Maximize size={16} /> : <Minimize size={16} />}
              </button>
              <button 
                onClick={toggleChat}
                className="text-gray-400 hover:text-white transition-colors"
              >
                <X size={16} />
              </button>
            </div>
          </div>
          
          {/* Chat messages */}
          {!isMinimized && (
            <>
              <div className="flex-1 p-4 overflow-y-auto">
                {messages.map((message, index) => (
                  <div key={index} className={`mb-3 ${message.sender === 'user' ? 'text-right' : ''}`}>
                    <div
                      className={`inline-block px-3 py-2 rounded-lg ${
                        message.sender === 'user'
                          ? 'bg-blue-600 text-white'
                          : 'bg-gray-800 text-gray-200'
                      }`}
                    >
                      {message.text}
                    </div>
                    {message.isGame && message.gameComponent && (
                      <div className="mt-2">{message.gameComponent}</div>
                    )}
                  </div>
                ))}
                <div ref={messagesEndRef}></div>
              </div>
              
              {/* Chat input */}
              <div className="p-3 border-t border-gray-700 bg-gray-800">
                <div className="flex items-center space-x-2">
                  <input
                    type="text"
                    value={inputValue}
                    onChange={(e) => setInputValue(e.target.value)}
                    onKeyPress={handleKeyPress}
                    placeholder="Type a message..."
                    className="flex-1 bg-gray-700 border border-gray-600 rounded-lg px-3 py-2 text-white focus:outline-none focus:border-blue-500"
                  />
                  <button
                    onClick={handleSendMessage}
                    className="bg-blue-600 hover:bg-blue-700 text-white p-2 rounded-lg transition-colors"
                  >
                    <Send size={18} />
                  </button>
                </div>
              </div>
            </>
          )}
        </div>
      )}
      
      {/* Chat button */}
      <button
        onClick={toggleChat}
        className="bg-blue-600 hover:bg-blue-700 text-white p-4 rounded-full shadow-lg transition-colors flex items-center justify-center"
      >
        {isOpen ? <X size={20} /> : <MessageCircle size={20} />}
      </button>
    </div>
  );
};

export default AIChatBox; 