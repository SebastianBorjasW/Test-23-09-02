import React from "react";
import ReactDOM from "react-dom/client";
import {createBrowserRouter, RouterProvider} from "react-router-dom";
import { path } from "@tauri-apps/api";
import Home from "./routes/home";


const router = createBrowserRouter([
  {
    path : "/",  
    element: <Home /> 
  }
]);



ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
      <RouterProvider router={router}/>
  </React.StrictMode>,
)
