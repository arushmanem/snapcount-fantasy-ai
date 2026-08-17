import { useState, useEffect, useRef } from 'react';

function PlayerSearch({ players, value, onChange, placeholder }) {
  const [isOpen, setIsOpen] = useState(false);
  const [filter, setFilter] = useState('');
  const wrapperRef = useRef(null);

  // Close dropdown if clicked outside
  useEffect(() => {
    function handleClickOutside(event) {
      if (wrapperRef.current && !wrapperRef.current.contains(event.target)) {
        setIsOpen(false);
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, [wrapperRef]);

  // Handle typing
  const handleInputChange = (e) => {
    setFilter(e.target.value);
    onChange(e.target.value);
    setIsOpen(true);
  };

  // Handle selecting an item
  const handleSelect = (player) => {
    onChange(player);
    setFilter(player);
    setIsOpen(false);
  };

  // Filter the list based on input
  const filteredPlayers = players.filter(p => 
    p.toLowerCase().includes(filter.toLowerCase())
  );

  return (
    <div ref={wrapperRef} style={{ position: 'relative', width: '220px' }}>
      <input
        type="text"
        value={value} // Controlled by parent
        onChange={handleInputChange}
        onFocus={() => setIsOpen(true)}
        placeholder={placeholder}
        style={{ width: '100%' }} // Ensure input fills container
      />
      
      {isOpen && filteredPlayers.length > 0 && (
        <ul style={{
          position: 'absolute',
          top: '100%',
          left: 0,
          right: 0,
          background: '#1e293b', // Matches var(--input-bg)
          border: '1px solid #334155',
          borderRadius: '12px',
          marginTop: '5px',
          maxHeight: '250px',
          overflowY: 'auto',
          listStyle: 'none',
          padding: '5px',
          margin: 0,
          zIndex: 1000,
          boxShadow: '0 10px 15px -3px rgba(0, 0, 0, 0.5)'
        }}>
          {filteredPlayers.map((player) => (
            <li 
              key={player}
              onClick={() => handleSelect(player)}
              style={{
                padding: '10px 15px',
                cursor: 'pointer',
                color: '#f8fafc',
                borderBottom: '1px solid rgba(255,255,255,0.05)',
                fontSize: '0.9rem',
                borderRadius: '8px',
                transition: 'background 0.2s'
              }}
              onMouseEnter={(e) => e.target.style.background = '#3b82f6'} // Hover Blue
              onMouseLeave={(e) => e.target.style.background = 'transparent'}
            >
              {player}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

export default PlayerSearch;