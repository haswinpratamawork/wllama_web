/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      // fontFamily: {
      //   poppins: ['Poppins', 'sans-serif'],
      //   inter: ['Inter', 'sans-serif'],
      // }
    },
  },
  plugins: [require('daisyui')],
  // daisyui: {
  //   themes: [
  //     {
  //       mytheme: {
  //         "primary": "#570df8",
  //         "secondary": "#f000b8",
  //         "accent": "#37cdbe",
  //         "neutral": "#0a378aff",
  //         "base-100": "#883333ff",
  //         "base-200": "#000000ff",
  //         "base-300": "#457ed2ff",
  //         "info": "#3abff8",
  //         "success": "#36d399",
  //         "warning": "#fbbd23",
  //         "error": "#f87272",
  //       },
  //     },
  //     "dark", // theme lain jika mau
  //   ],
  // },
};