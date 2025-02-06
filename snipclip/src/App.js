import './App.css';
import 'bootstrap/dist/css/bootstrap.min.css';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { Home } from './screens/Home';
import NavBarElements from './components/NavBar/NavBarElements';
import { About } from './screens/About';

function App() {
  return (
    <Router>
      <NavBarElements />
      <Routes>
        <Route path='/' element={<Home />} />
        <Route path='/About' element={<About />} />
      </Routes>
    </Router>
  );
}

export default App;
