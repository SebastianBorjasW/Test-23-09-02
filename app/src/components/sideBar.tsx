import { Link } from "react-router-dom";

export default function Sidebar() {
  return (
    <div className="h-screen bg-gray-800 text-white w-60 flex flex-col justify-between">
      <div>
        {/* Logo */}
        <div className="p-4 text-center font-bold text-lg">
          <h1>Image</h1>
          
          <h1>classifier</h1>
        </div>
        

        {/* Links */}
        <nav className="flex flex-col p-4 space-y-4">
          <Link
            to="/"
            className="hover:bg-gray-700 p-2 rounded transition-colors"
          >
            Home
          </Link>
          <Link
            to="/statistics"
            className="hover:bg-gray-700 p-2 rounded transition-colors"
          >
            Statistics
          </Link>
        </nav>
      </div>

      <div className="p-4">
        <p>© 2024 GAON Test</p>
      </div>
    </div>
  );
}
