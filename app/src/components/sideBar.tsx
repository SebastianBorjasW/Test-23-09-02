import { useState } from "react";
import { Link } from "react-router-dom";

export default function Sidebar() {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <div className="flex h-screen">
      {/* Botón para abrir/cerrar el menú en pantallas pequeñas */}
      <div className="md:hidden flex items-center p-4">
        <button
          onClick={() => setIsOpen(!isOpen)}
          className="text-white focus:outline-none"
        >
          <svg
            className="w-6 h-6"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
            xmlns="http://www.w3.org/2000/svg"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M4 6h16M4 12h16M4 18h16"
            />
          </svg>
        </button>
      </div>

      {/* Barra lateral que se ajusta en tamaño dependiendo del viewport */}
      <div
        className={`${
          isOpen ? "block" : ""
        } h-screen bg-gray-800 text-white flex flex-col justify-between md:relative fixed inset-y-0 left-0 z-50 transition-all duration-300
         w-20 md:w-60`}
      >
        <div>
          {/* Logo, ajustar el tamaño según el viewport */}
          <div className="p-4 text-center font-bold text-lg md:text-xl transition-all duration-300">
            <h1 className="truncate">Img</h1>
            <h1 className="truncate md:block hidden">Classifier</h1>
          </div>

          {/* Links */}
          <nav className="flex flex-col p-4 space-y-4">
            <Link
              to="/"
              className="hover:bg-gray-700 p-2 rounded transition-colors text-center"
              onClick={() => setIsOpen(false)}
            >
              <span className="block md:hidden">🏠</span>
              <span className="hidden md:block">Home</span>
            </Link>
            <Link
              to="/statistics"
              className="hover:bg-gray-700 p-2 rounded transition-colors text-center"
              onClick={() => setIsOpen(false)}
            >
              <span className="block md:hidden">📊</span>
              <span className="hidden md:block">Statistics</span>
            </Link>
          </nav>
        </div>

        <div className="p-4 mt-auto">
          <p className="hidden md:block">© 2024 GAON Test</p>
        </div>
      </div>

      {/* Capa de fondo cuando el menú está abierto en dispositivos pequeños */}
      {isOpen && (
        <div
          className="fixed inset-0 bg-black opacity-50 z-40"
          onClick={() => setIsOpen(false)}
        ></div>
      )}
    </div>
  );
}
