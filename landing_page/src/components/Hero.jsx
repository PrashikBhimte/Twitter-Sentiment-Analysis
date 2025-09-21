import React from 'react';

const Hero = () => {
  return (
    <section className="bg-gray-900 text-white text-center py-20">
      <div className="container mx-auto">
        <h2 className="text-5xl font-bold mb-4">Powerful Tweet Classification, Right from Your Terminal</h2>
        <p className="text-xl mb-8">Leverage pre-trained models to instantly analyze sentiment and classify tweets with a simple command.</p>
        <a href="/tweet_classifier_cli.zip" className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-3 px-6 rounded-full text-lg" download>
          Download Tool
        </a>
      </div>
    </section>
  );
};

export default Hero;
