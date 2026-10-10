import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
import LloApp from './llo/LloApp';
import './styles.css';

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>{new URLSearchParams(location.search).get('view') === 'llo' ? <LloApp /> : <App />}</React.StrictMode>
);

if (import.meta.env.PROD && 'serviceWorker' in navigator) {
  window.addEventListener('load', () => {
    void navigator.serviceWorker.register(`${import.meta.env.BASE_URL}sw.js`).catch(error => {
      console.warn('PWA offline cache unavailable:', error);
    });
  });
}
