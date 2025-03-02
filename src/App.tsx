import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Navbar from './components/Navbar';
import Footer from './components/Footer';
import Home from './pages/Home';
import Record from './pages/Record';
import Dashboard from './pages/Dashboard';
import Masterclass from './pages/Masterclass';
import MasterclassDetail from './pages/MasterclassDetail';
import Schedule from './pages/Schedule';
import NotFound from './pages/NotFound';

function App() {
  return (
    <Router>
      <div className="flex flex-col min-h-screen">
        <Navbar />
        <main className="flex-grow">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/record" element={<Record />} />
            <Route path="/dashboard" element={<Dashboard />} />
            <Route path="/masterclass" element={<Masterclass />} />
            <Route path="/masterclass/:id" element={<MasterclassDetail />} />
            <Route path="/schedule" element={<Schedule />} />
            <Route path="*" element={<NotFound />} />
          </Routes>
        </main>
        <Footer />
      </div>
    </Router>
  );
}

export default App;