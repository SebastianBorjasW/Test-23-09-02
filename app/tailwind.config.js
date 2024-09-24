/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
				background: 'black',
				'light-background': '#191919',
				'lighter-background': '#2c2c2c',
				'text-primary': 'white',
				'text-secondary': '#b3b3b3',
				'text-accent': '#c29f11',
			},
    },
  },
  plugins: [],
}

