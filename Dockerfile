FROM node:18-alpine
WORKDIR /app
COPY package.json package-lock.json* ./
COPY wllama-wllama-2.3.5.tgz ./
RUN npm ci
COPY . .
RUN npm run build
RUN npm install -g serve
EXPOSE 3000
CMD ["serve", "-s", "dist", "-l", "3000"]
