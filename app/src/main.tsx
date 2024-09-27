import React from "react";
import ReactDOM from "react-dom/client";
import {createBrowserRouter, RouterProvider} from "react-router-dom";
import Home from "./routes/home";
import Statistics from "./routes/statistics";
import "./index.css";


const router = createBrowserRouter([
  {
    path : "/",  
    element: <Home /> 
  },
  {
    path : "/statistics",
    element: <Statistics />
  }
]);



ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
      <RouterProvider router={router}/>
  </React.StrictMode>,
)
