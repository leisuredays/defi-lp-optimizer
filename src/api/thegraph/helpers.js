// Subgraph IDs for Uniswap V3 on The Graph's decentralized network
// Source: https://docs.uniswap.org/api/subgraph/overview
const SUBGRAPH_IDS = {
  0: '5zvR82QoaXYFyDEKLZ9t6v9adgnptxYpKpSbxtgVENFV', // Ethereum Mainnet
  1: 'Cghf4LfVqPiFw6fp6Y5X5Ubc8UpmUhSfJL82zwiBFLaj', // Optimism
  2: 'FbCGRftH4a3yZugY7TnbYgPJVEv2LvMT6oF1fxPe9aJM', // Arbitrum
  3: '3hCPRGf4z88VC5rsBKU5AA9FBBq5nF3jbKJG7VZCbhjm', // Polygon
  4: 'Cghf4LfVqPiFw6fp6Y5X5Ubc8UpmUhSfJL82zwiBFLaj', // Perpetual (using Optimism for now)
  5: 'ESdrTJ3twMwWVoQ1hUE2u7PugEHX3QkenudD6aXCkDQ4', // Celo
};

export const urlForProtocol = (protocol) => {
  const apiKey = process.env.REACT_APP_GRAPH_API_KEY;

  if (!apiKey) {
    console.error('REACT_APP_GRAPH_API_KEY is not set. Please add it to your .env file.');
    console.error('Get your API key from: https://thegraph.com/studio/');
    // Fallback to old hosted service (deprecated but may still work temporarily)
    return protocol === 1 ? "https://api.thegraph.com/subgraphs/name/ianlapham/optimism-post-regenesis" :
      protocol === 2 ? "https://api.thegraph.com/subgraphs/name/ianlapham/arbitrum-minimal" :
      protocol === 3 ? "https://api.thegraph.com/subgraphs/name/ianlapham/uniswap-v3-polygon" :
      protocol === 4 ? "https://api.thegraph.com/subgraphs/name/perpetual-protocol/perpetual-v2-optimism" :
      protocol === 5 ? "https://api.thegraph.com/subgraphs/name/jesse-sawa/uniswap-celo" :
      "https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v3";
  }

  const subgraphId = SUBGRAPH_IDS[protocol] || SUBGRAPH_IDS[0];
  return `https://gateway.thegraph.com/api/${apiKey}/subgraphs/id/${subgraphId}`;
}

export const minTvl = (protocol) => {
  return protocol === 0 ? 10000 : 1;
}

export const requestBody = (request) => {
  
  if(!request.query) return;

  const body = {
      method:'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ 
        query: request.query,
        variables: request.variables || {}
      })
  }

  if (request.signal) body.signal = request.signal;
  return body;

}


