import React from 'react';

const Features = () => {
  const features = [
    {
      title: 'Instant Analysis',
      description: 'Fast and reliable predictions.',
    },
    {
      title: 'Model Switching',
      description: 'Easily switch between different trained models.',
    },
    {
      title: 'Simple Installation',
      description: 'Packaged for easy setup with pip.',
    },
  ];

  return (
    <section className="bg-gray-800 text-white py-20">
      <div className="container mx-auto text-center">
        <h3 className="text-3xl font-bold mb-12">Features</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          {features.map((feature, index) => (
            <div key={index} className="bg-gray-700 p-6 rounded-lg">
              <h4 className="text-xl font-bold mb-2">{feature.title}</h4>
              <p>{feature.description}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default Features;
