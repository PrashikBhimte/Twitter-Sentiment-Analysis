import React from 'react';
import Header from './components/Header';
import Hero from './components/Hero';
import Features from './components/Features';
import Usage from './components/Usage';
import Footer from './components/Footer.jsx';
import Report from './components/Report.jsx';

const App = () => {
  return (
    <div className="bg-gray-900 min-h-screen">
      <Header />
      <main>
        <Hero />
        <Features />
        <Usage />
        <Report />
      </main>
      <Footer />
    </div>
  );
};

export default App;
