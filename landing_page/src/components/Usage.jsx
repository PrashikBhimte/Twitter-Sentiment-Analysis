import React from 'react';

const Usage = () => {
  const codeString = `
# Install the tool from the local package
pip install .

# Run a prediction
classify-tweet "The new update is amazing!"

# Use a different model
classify-tweet "I'm not sure about these changes..." -m model_b
  `;

  return (
    <section className="bg-gray-900 text-white py-20">
      <div className="container mx-auto">
        <h3 className="text-3xl font-bold text-center mb-12">How to Use</h3>
        <div className="bg-gray-800 rounded-lg p-6 max-w-2xl mx-auto">
          <pre className="text-left text-sm font-mono whitespace-pre-wrap">
            <code>{codeString}</code>
          </pre>
        </div>
      </div>
    </section>
  );
};

export default Usage;
