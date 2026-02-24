import { useState } from 'react';
import './App.css';
import TryOn from './TryOn';
import NavBar from './NavBar';
import Suggestion from './Suggestion';
import { Route, Routes } from "react-router-dom";

function App() {
  const [page, setPage] = useState('tryon')

  return (
    <div>
      <NavBar/>

      <div>
        <Routes>
          <Route path="/" element={<TryOn />} />
          <Route path="/tryon" element={<TryOn />} />
          <Route path="/suggestion" element={<Suggestion />} />
        </Routes>
      </div>

    </div>
  );
}

export default App;
